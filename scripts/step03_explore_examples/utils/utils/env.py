import os


def check_env_variable(env_variable: str):
    """
    Checks if the specified environment variable is set.

    Args:
            env_variable (str): The name of the environment variable to check.

    Raises:
            ValueError: If the environment variable is not set.

    """
    if env_variable not in os.environ:
        raise ValueError(f"Environment variable {env_variable} is not set.")


def assert_env(env_variables: list):
    """
    Asserts that the specified environment variables are set.

    Args:
            env_variables (list): A list of environment variable names to check.

    Raises:
            ValueError: If any of the specified environment variables are not set.

    Returns:
            None
    """

    for env_variable in env_variables:
        check_env_variable(env_variable)
