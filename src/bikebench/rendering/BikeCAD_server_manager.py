import atexit
import os
import subprocess
import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import Future
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List

import psutil
import requests

from bikebench.exceptions import InternalError, check_internal_precondition
from bikebench.resource_utils import resource_path


def get_java_binary():
    b = os.getenv("JAVA_HOME", "java")
    if b.endswith("java"):
        res = b
    else:
        res = os.path.join(b, "bin", "java")
    print(f"Using {res} as the Java binary")
    return res


JAVA_BINARY = get_java_binary()


class ServerManager(metaclass=ABCMeta):
    def __init__(self):
        self._server_pids: List[int] = []
        atexit.register(self._kill_live_servers)

    @abstractmethod
    def endpoint(self, suffix: str) -> str:
        pass

    def _start_server(self, port: int, timeout_seconds: int) -> None:
        if not self._check_server_health(port):
            print(f"Starting BikeCAD server on port {port}...")
            process = subprocess.Popen(
                [JAVA_BINARY, "-jar", resource_path("BikeCAD-server.jar"), f"--server.port={port}"])
            self._server_pids.append(process.pid)
            self._await_start_or_throw(port, timeout_seconds)
            print(f"BikeCAD server started on port {port}.")

    def _await_start_or_throw(self,
                              port: int,
                              timeout_seconds: int
                              ) -> None:
        seconds_waited = 0
        while not self._check_server_health(port):
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited > timeout_seconds:
                raise InternalError(f"Could not start server on port {port}...")

    def _kill_live_servers(self) -> None:
        for server_pid in self._server_pids:
            if psutil.pid_exists(server_pid):
                print(f"BikeCAD Server with pid {server_pid} exists. Killing...")
                psutil.Process(pid=server_pid).kill()
                print(f"BikeCAD server with pid {server_pid} killed successfully.")
            else:
                print(f"WARNING: pid {server_pid} does not exist")

    def _check_server_health(self, port: int) -> bool:
        try:
            health_response = requests.get(self._endpoint(port, "/actuator/serverInformation"), timeout=1)
        except Exception as ignored:
            return False
        if health_response.status_code != 200:
            return False
        return health_response.json()["serverName"] == "BikeCAD-server"

    def _endpoint(self, port: int, suffix: str):
        url = f"http://localhost:{port}{suffix}"
        check_internal_precondition(suffix.startswith("/"), f"Invalid url {url}")
        return url


class SingleThreadedBikeCadServerManager(ServerManager):
    SERVER_PORT = 8080

    def __init__(self, timeout_seconds: int):
        super().__init__()
        self._start_server(self.SERVER_PORT, timeout_seconds)

    def endpoint(self, suffix: str) -> str:
        return self._endpoint(self.SERVER_PORT, suffix)


class MultiThreadedBikeCadServerManager(ServerManager):
    STARTING_PORT = 8080

    def __init__(self, number_servers: int, timeout_seconds: int):
        super().__init__()
        self._port_range = [self.STARTING_PORT + i for i in range(number_servers)]
        self._request_count = 0  # used for round-robin load-balancing :D
        futures = self._start_servers(number_servers, timeout_seconds)
        self._await_servers(futures, timeout_seconds)

    def _start_servers(self, number_servers: int, timeout_seconds: int) -> List[Future]:
        executor = ThreadPoolExecutor(max_workers=number_servers)
        futures = []
        for port in self._port_range:
            futures.append(executor.submit(self._start_server, port, timeout_seconds))
        return futures

    def endpoint(self, suffix: str) -> str:
        n_servers = len(self._port_range)
        selected_port = self._port_range[self._request_count % n_servers]  # modulo for safety

        self._update_request_count(n_servers)
        return self._endpoint(
            selected_port,
            suffix
        )

    def _update_request_count(self, n_servers: int):
        self._request_count += 1
        if self._request_count >= n_servers:
            self._request_count = 0  # restart counter

    def _await_servers(self, futures: List[Future], timeout_seconds: int):
        seconds_waited = 0
        while not all([f.done() for f in futures]):
            time.sleep(1)
            seconds_waited += 1
            if seconds_waited >= timeout_seconds:
                raise InternalError("Failed to start servers in time")
