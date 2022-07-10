#!/usr/bin/env python3

import subprocess


def revision():
    """Return string with current revision"""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("UTF-8").strip()


def branch():
    """Return string with current branch name"""
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("UTF-8").strip()


def dateOfLastRevision():
    """Last revision date information"""
    return subprocess.check_output(["git", "log", "-n1", "--date=short", "--pretty=format:%cd"]).decode("UTF-8").replace("-", ".").strip()


def dateAndTimeOfLastRevision():
    """Last revision date and time information"""
    return subprocess.check_output(["git", "log", "-n1", "--date=short", "--pretty=format:%cD"]).decode("UTF-8").replace("-", ".").strip()
