/*
 * socket_utils.c
 * --------------
 * Thin C wrappers around POSIX sockets for use from Fortran via ISO_C_BINDING.
 * Compile with:  gcc -O2 -c socket_utils.c -o socket_utils.o
 */

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>

/* macOS has no MSG_NOSIGNAL; SO_NOSIGPIPE on the socket does the same job */
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

/* ── connect ──────────────────────────────────────────────────────────────── */

/**
 * sock_connect_c(host_c, port) -> file descriptor, or -1 on error
 * host_c must be a null-terminated ASCII string (e.g. "127.0.0.1")
 */
int sock_connect_c(const char *host_c, int port)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }

    /* disable Nagle for low-latency request/response */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    /* servers spawned later must not inherit this connection */
    fcntl(fd, F_SETFD, FD_CLOEXEC);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (inet_pton(AF_INET, host_c, &addr.sin_addr) <= 0) {
        fprintf(stderr, "inet_pton failed for '%s'\n", host_c);
        close(fd);
        return -1;
    }

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(fd);
        return -1;
    }
    return fd;
}

/* ── send / recv (blocking, exactly n bytes) ─────────────────────────────── */

/**
 * Returns 0 on success, -1 on error/disconnect.
 */
int sock_send_c(int fd, const void *buf, int nbytes)
{
    const char *p = (const char *)buf;
    int remaining = nbytes;
    while (remaining > 0) {
        ssize_t n = send(fd, p, (size_t)remaining, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0) perror("send");
            return -1;
        }
        p         += n;
        remaining -= (int)n;
    }
    return 0;
}

int sock_recv_c(int fd, void *buf, int nbytes)
{
    char *p = (char *)buf;
    int remaining = nbytes;
    while (remaining > 0) {
        ssize_t n = recv(fd, p, (size_t)remaining, MSG_WAITALL);
        if (n <= 0) {
            if (n < 0) perror("recv");
            return -1;
        }
        p         += n;
        remaining -= (int)n;
    }
    return 0;
}

/* ── close ───────────────────────────────────────────────────────────────── */

void sock_close_c(int fd)
{
    if (fd >= 0) close(fd);
}

/* ── process management ──────────────────────────────────────────────────── */

/**
 * Spawn a command in a new process group.
 * Returns the child PID, or -1 on error.
 * stdout_fd_out is set to the read end of a pipe connected to child stdout,
 * so the caller can wait for the "READY" line.
 */
int spawn_process_c(const char *cmd, int *stdout_fd_out)
{
    int pipefd[2];
    if (pipe(pipefd) < 0) { perror("pipe"); return -1; }

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        /* child */
        close(pipefd[0]);                    /* close read end */
        dup2(pipefd[1], STDOUT_FILENO);      /* stdout -> pipe write end */
        close(pipefd[1]);
        /* new session so we can kill the whole group */
        setsid();
        execl("/bin/sh", "sh", "-c", cmd, (char *)NULL);
        _exit(127);
    }

    /* parent */
    close(pipefd[1]);          /* close write end */
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    *stdout_fd_out = pipefd[0];
    return (int)pid;
}

/**
 * Read from fd until a line matching prefix is found (or error/EOF).
 * Returns parsed port (>0) for "READY <port>", 0 for plain "READY", -1 on timeout/error.
 */
int wait_for_line_c(int fd, const char *prefix, int timeout_sec)
{
    /* set non-blocking read with simple timeout loop */
    char buf[512];
    int  pos = 0;
    time_t start_time = time(NULL);
    time_t deadline = start_time + timeout_sec;
    time_t now;
    time_t next_report = start_time + 60;
    int   plen = (int)strlen(prefix);
    int old_flags;
    char *endptr;
    long parsed_port;

    old_flags = fcntl(fd, F_GETFL, 0);
    if (old_flags < 0) return -1;

    if (fcntl(fd, F_SETFL, old_flags | O_NONBLOCK) < 0) return -1;

    while ((now = time(NULL)) < deadline) {
        if (now >= next_report) {
            fprintf(stdout,
                    "mlip_init: still waiting for READY signal (%ld s elapsed)\n",
                    (long)(now - start_time));
            fflush(stdout);
            next_report += 60;
        }
        ssize_t n = read(fd, buf + pos, 1);
        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                usleep(10000); /* 10 ms */
                continue;
            }
            fcntl(fd, F_SETFL, old_flags);
            return -1;
        }
        if (n == 0) {
            fcntl(fd, F_SETFL, old_flags);
            return -1; /* EOF */
        }
        if (buf[pos] == '\n') {
            buf[pos] = '\0';
            if (strncmp(buf, prefix, (size_t)plen) == 0) {
                if (buf[plen] == '\0') {
                    fcntl(fd, F_SETFL, old_flags);
                    return 0;
                }
                if (buf[plen] == ' ') {
                    parsed_port = strtol(buf + plen + 1, &endptr, 10);
                    if ((endptr != buf + plen + 1) && (*endptr == '\0') &&
                        (parsed_port > 0) && (parsed_port <= 65535)) {
                        fcntl(fd, F_SETFL, old_flags);
                        return (int)parsed_port;
                    }
                }
                fcntl(fd, F_SETFL, old_flags);
                return 0;
            }
            pos = 0; /* next line */
        } else {
            pos++;
            if (pos >= 511) pos = 511; /* guard */
        }
    }
    fcntl(fd, F_SETFL, old_flags);
    return -1; /* timeout */
}

/**
 * Send SIGTERM to the process group led by pid and reap the child.
 * spawn_process_c() runs the command via "sh -c" in a new session, so pid
 * is the shell; if the shell forked instead of exec'ing, the server is a
 * child and only a group kill reaches it.  Falls back to pid alone.
 *
 * The child is waited for, so it does not stay behind as a zombie for the
 * lifetime of the caller.  A server that has not exited after 10 s is
 * killed with SIGKILL.
 */
void kill_process_c(int pid)
{
    int i, status;
    pid_t r;

    if (pid <= 0) return;
    if (kill(-(pid_t)pid, SIGTERM) < 0) kill((pid_t)pid, SIGTERM);

    for (i = 0; i < 1000; i++) {
        r = waitpid((pid_t)pid, &status, WNOHANG);
        if (r != 0) return;     /* reaped, or not our child (r < 0) */
        usleep(10000);          /* 10 ms */
    }
    if (kill(-(pid_t)pid, SIGKILL) < 0) kill((pid_t)pid, SIGKILL);
    waitpid((pid_t)pid, &status, 0);
}


/**
 * Thin shim exposing getpid() with a plain C int return type.
 * Fortran's iso_c_binding can map this directly via c_int.
 * Cast is explicit to silence any implicit-narrowing warnings
 * (pid_t is typically a 32-bit int on modern platforms anyway). 
 */
int get_pid_shim(void) {
    return (int)getpid();
}
