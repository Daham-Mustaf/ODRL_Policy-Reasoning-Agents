human text → reasoning → structured output → saving for future use.




## Configuration

This project uses a configuration file to manage Azure OpenAI API settings. 

### Setting up configuration

1. Copy `config.template.json` to `config.json`
2. Edit `config.json` and add your Azure OpenAI API credentials
3. Set the `active_model` to your preferred model

Alternatively, you can use environment variables:
- `AZURE_API_KEY`: Your Azure OpenAI API key
- `AZURE_ENDPOINT`: Your Azure OpenAI endpoint URL
- `AZURE_API_VERSION`: API version to use
- `DEPLOYMENT_NAME`: Name of the model deployment to use