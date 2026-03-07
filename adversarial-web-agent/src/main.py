import yaml
import logging

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(f"Starting {config['agent']['name']} v{config['agent']['version']}...")

if __name__ == "__main__":
    main()
