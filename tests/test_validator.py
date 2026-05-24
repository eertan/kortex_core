import pytest
from kortex.sandbox.validator import SecurityValidator

def test_security_validator():
    validator = SecurityValidator()
    
    # 1. Safe Code
    safe_code = """
from kortex.plugins.registry import registry
@registry.register_action("charge_battery")
def charge(robot: str) -> str:
    return f"{robot} is charging"
"""
    assert validator.validate_code(safe_code) == True
    
    # 2. Banned Import
    unsafe_import = """
import os
os.system("rm -rf /")
"""
    assert validator.validate_code(unsafe_import) == False
    
    # 3. Banned Call (eval)
    unsafe_call = """
def execute_dynamic(cmd):
    return eval(cmd)
"""
    assert validator.validate_code(unsafe_call) == False
