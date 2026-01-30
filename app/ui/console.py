from rich.console import Console
from rich.text import Text

console = Console()

def print_event(event):
    if not event.is_final:
        console.print(event.text, end="")
    else:
        console.print()
