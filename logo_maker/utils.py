import msvcrt
import sys
import select


def q_key_pressed_non_blocking() -> bool:
    input_key: str | None = None

    if sys.platform == "win32":
        if msvcrt.kbhit():
            # Read the key if there is any
            input_key = msvcrt.getch().decode()
    else:
        # Check if there is any input waiting to be read without blocking
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            # Read the input if there is any
            input_key = sys.read(sys.stdin.fileno(), 1)

    if input_key == "q":
        print("\n pressed 'q' key ...\n")
        return True
    else:
        return False





