import click


@click.group()
@click.version_option()
def cli() -> None:
    pass


@cli.command()
def hello() -> None:
    click.echo("Hello World!")
