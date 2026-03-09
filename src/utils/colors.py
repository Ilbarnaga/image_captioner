from colorama import Fore, Back, Style, init

# Initialize colorama globally for the whole app
init(autoreset=True)

# Map the old names to Colorama constants
RED         = Fore.RED
GREEN       = Fore.GREEN
YELLOW      = Fore.YELLOW
CYAN        = Fore.CYAN
VIOLET      = Fore.MAGENTA
VIOLET_BACK = Back.MAGENTA
ORANGE      = '\033[38;2;255;165;0m'
BOLD        = Style.BRIGHT
NC          = Style.RESET_ALL