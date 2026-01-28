## Run Code
To run packages we need to use the following terminal command:

```bash
python3 -m package
```

For example we have the packages `experiments` and `synergygrid` at the moment. This way all imports point to the right places.

## Debug Code
We also have a launch.json in our .vscode folder that enables normal debugging through VS Code. We just need to specify what package to debug in the sidebar. And if we need to debug specified packages or modules in the future, just add another configuration in the JSON file.