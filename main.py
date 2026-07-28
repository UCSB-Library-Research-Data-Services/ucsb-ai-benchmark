from openai import OpenAI
import logging        
import argparse
from pprint import pp
from config import SERVICES
import sys

def _client(service: dict) -> OpenAI:
    """helper function to build the client object"""
    if not isinstance(service, dict):
        raise ValueError(f"Expected a sevice config dict, got: {type(service)}")
    return OpenAI(
        base_url=service.get("url"),
        api_key=service.get("key"),
        timeout=120.0
    )

def list_models_from_service(service_name: str, service: dict, verbose: bool =False) -> None:
    
   client = _client(service)
   models_page = client.models.list()
   print(f"\nModels for {service_name}:")
   for model in models_page.data:
       if verbose:
           pp(model.model_dump(), indent=2)
       else:
           print(f"Model ID: {model.id}")
    

def list_models(verbose: bool =False) -> None:
    for name, config in SERVICES.items():
        if not config.get("key"):
            logging.warning(f"Skipping '{name}: missing API key")
            continue
        try:
            list_models_from_service(name, config, verbose=verbose)
        except Exception as e:
            logging.error(f"Error listing models for {name}: {e}")
            
def _validate_service(parser: argparse.ArgumentParser, service_name: str) -> dict:
    if service_name not in SERVICES:
        parser.error(f"Unknown service '{service_name}'. Available: {list(SERVICES)}")
    config = SERVICES[service_name]
    if not config.get("key"):
        parser.error(f"Service '{service_name}' has no API key configured")
    return config            

def main():
    parser = argparse.ArgumentParser(description="Basic CLI to interact with UCSB LLM providers")
    
    list_group = parser.add_mutually_exclusive_group()
    
    parser.add_argument("--service", type=str, help=f"Service provider (e.g. 'grit', 'dream-lab'). " f"Available: {list(SERVICES)}")
    list_group.add_argument("--list-models", action="store_true", help="List model IDs. Scoped to --service if provided, otherwise all services.")
    list_group.add_argument("--list-models-v", action="store_true", help="Like --list-models but dumps full model metadata.")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.list_models or args.list_models_v:
        verbose = args.list_models_v
        if isinstance(args.service, str):
            service_config = _validate_service(parser, args.service)
            list_models_from_service(args.service, service_config, verbose=verbose)
        else:
            list_models(verbose=verbose)
        return

if __name__ == "__main__":
    main()
