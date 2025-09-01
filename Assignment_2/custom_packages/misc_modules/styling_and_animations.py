#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 12:23:58 2025

@author: constantina
"""

from custom_packages.misc_modules import config
import time
import sys

def print_error_message(message: str, e: Exception) -> None:
    """Displaying a red-coloured error message in the console. """
    print(f"{config.ANSI_LIGHT_RED}{message}:{config.ANSI_RESET_COLOUR}", {e})

def print_success_message(message: str) -> None:
    """Displaying a green-coloured success message in the console. """
    print(f"{config.ANSI_LIGHT_GREEN}{message}{config.ANSI_RESET_COLOUR}")

def print_warning_message(message: str) -> None:
    """Displaying an orange-coloured warning message in the console. """
    print(f"{config.ANSI_LIGHT_ORANGE}{message}{config.ANSI_RESET_COLOUR}")

def print_highlighted_message(message: str) -> None:
    """Displaying a yellow-highlighted message in the console. """
    print(f"{config.ANSI_BACKGROUND_LIGHT_YELLOW}{config.ANSI_BLACK}{message}{config.ANSI_RESET_COLOUR}")


def print_animated_ellipsis_message(message: str ="Currently processing", duration: float = 4.0, sleep_interval: float = 0.2) -> None:
    """
    Displaying a message in the console, followed by a simple, animated ellipsis to indicate progress.

    Parameters
    ----------
    message : str, optional
        The message to be displayed in the console. 
        The default is "Currently processing".
    duration : float, optional
        Indicates the total duration (seconds) of the animated ellipsis. 
        The default is 4.0.
    sleep_interval : float, optional
        Indicating how many seconds needed before each ellipsis update. 
        The default is 0.2.

    Returns
    -------
    None

    """
    
    ellipsis = [" ", ".", "..", "..."]
    end_time = time.time() + duration
    
    i = 0
    sys.stdout.write("\n")
    while time.time() < end_time:
        sys.stdout.write(f"\r~~~~~~~~~ {message}{ellipsis[i % len(ellipsis)]} ~~~~~~~~~")
        sys.stdout.flush()
        time.sleep(sleep_interval)
        i += 1
        
    sys.stdout.write("\n\n")
    
    return None

