import ast

class SecurityValidator:
    """
    Validates auto-generated python plugins before they are allowed into the registry.
    """
    
    BANNED_IMPORTS = {"os", "sys", "subprocess", "socket", "requests", "shutil"}
    BANNED_CALLS = {"eval", "exec", "open"}
    
    def validate_code(self, code_string: str) -> bool:
        """
        Parses the AST of the provided python code and checks for malicious or
        banned operations. Returns True if safe, False if dangerous.
        """
        try:
            tree = ast.parse(code_string)
        except SyntaxError:
            print("[Validator] Syntax Error in generated code.")
            return False
            
        for node in ast.walk(tree):
            # Check Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.BANNED_IMPORTS:
                        print(f"[Validator] Banned import detected: {alias.name}")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module in self.BANNED_IMPORTS:
                    print(f"[Validator] Banned import detected: {node.module}")
                    return False
                    
            # Check Function Calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BANNED_CALLS:
                        print(f"[Validator] Banned function call detected: {node.func.id}")
                        return False
                        
        return True
