# Development

The `metalearners` repository can be cloned as follows:

```bash
git clone https://github.com/Quantco/metalearners.git
```

The dependencies are managed via [`pixi`](https://pixi.sh/latest/). Please make sure to install `pixi` on your system. Once `pixi` is installed, you can install and run the pre-commit hooks as follows:

```bash
pixi run pre-commit-install
pixi run pre-commit-run
```

You can run the tests as follows:

```bash
pixi run postinstall
pixi run pytest tests
```

You can preview the documentation locally by running:

```bash
pixi run -e docs postinstall
pixi run -e docs docs
```

Mkdocs will start a local server and you can preview the documentation by visiting `http://localhost:8000`.

You can find all `pixi` tasks in the `pixi.toml` file.
