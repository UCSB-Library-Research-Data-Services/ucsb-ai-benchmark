import json
import os
import logging
from datetime import datetime

class TokenTracker:
    def __init__(self, storage_file: str = ".token_usage.json"):
        self.storage_file = storage_file
        self.usage_data = self._load_data()

    def _load_data(self) -> dict:
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse {self.storage_file}, starting fresh.")
                return {}
        return {}

    def _save_data(self) -> None:
        with open(self.storage_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)

    def _get_current_month_key(self) -> str:
        """Returns the current month in YYYY-MM format to group usage."""
        return datetime.now().strftime("%Y-%m")

    def add_usage(self, service: str, model: str, usage_obj) -> None:
        """
        Takes an OpenAI Usage object and records the tokens.
        usage_obj is typically expected to have .prompt_tokens and .completion_tokens
        """
        if not usage_obj:
            return

        month_key = self._get_current_month_key()
        
        if month_key not in self.usage_data:
            self.usage_data[month_key] = {}
            
        if service not in self.usage_data[month_key]:
            self.usage_data[month_key][service] = {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "models": {}
            }

        srv_data = self.usage_data[month_key][service]
        
        if model not in srv_data["models"]:
            srv_data["models"][model] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Extract values depending on whether usage_obj is a dict or an object
        prompt_tokens = getattr(usage_obj, 'prompt_tokens', 0) if not isinstance(usage_obj, dict) else usage_obj.get('prompt_tokens', 0)
        completion_tokens = getattr(usage_obj, 'completion_tokens', 0) if not isinstance(usage_obj, dict) else usage_obj.get('completion_tokens', 0)
        total_tokens = getattr(usage_obj, 'total_tokens', 0) if not isinstance(usage_obj, dict) else usage_obj.get('total_tokens', 0)

        # Update service totals
        srv_data["total_prompt_tokens"] += prompt_tokens
        srv_data["total_completion_tokens"] += completion_tokens
        srv_data["total_tokens"] += total_tokens

        # Update model totals
        srv_data["models"][model]["prompt_tokens"] += prompt_tokens
        srv_data["models"][model]["completion_tokens"] += completion_tokens
        srv_data["models"][model]["total_tokens"] += total_tokens

        self._save_data()

    def get_service_usage(self, service: str) -> int:
        """Returns the total tokens used by a service in the current month."""
        month_key = self._get_current_month_key()
        if month_key in self.usage_data and service in self.usage_data[month_key]:
            return self.usage_data[month_key][service].get("total_tokens", 0)
        return 0

    def check_budget(self, service: str, budget_limit: int) -> bool:
        """Returns True if the service is within its budget, False otherwise."""
        current_usage = self.get_service_usage(service)
        if current_usage >= budget_limit:
            logging.error(f"BUDGET EXCEEDED: Service '{service}' has used {current_usage} tokens. Limit is {budget_limit}.")
            return False
        return True
