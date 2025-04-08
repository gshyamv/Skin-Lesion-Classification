import socket
import subprocess

def ping_localhost_with_port_and_execute(port, fallback_command):
    host = "localhost"
    try:
        # Attempt to connect to the specified port
        with socket.create_connection((host, port), timeout=5) as sock:
            print(f"Successfully connected to {host} on port {port}.")
    except (socket.timeout, ConnectionRefusedError):
        print(f"Connection to {host} on port {port} failed. Executing fallback command.")
        # Run the static fallback command in a non-blocking way
        try:
            subprocess.Popen(fallback_command, shell=True)
            print("Fallback command started.")
        except Exception as e:
            print(f"An error occurred while executing the fallback command: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def kill_ollama():
    subprocess.Popen('ollama stop llama3.1:8b',shell=True)
    print("Ollama killed")

if __name__ == "__main__":
    # Specify the target port and fallback command
    target_port = 11434  # Replace with your desired port
    fallback_cmd = "ollama serve"  # Replace with your static fallback command
    ping_localhost_with_port_and_execute(target_port, fallback_cmd)
