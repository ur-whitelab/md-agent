import difflib
import functools

from pydantic import BaseModel


def validate_func_args(args_schema=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            if not args_schema:  # use for functions without args_schema
                num_args = func.__code__.co_argcount
                varnames = list(func.__code__.co_varnames)
                valid_args = varnames[:num_args]
            elif issubclass(args_schema, BaseModel):
                print("duck")
                valid_args = list(args_schema.model_json_schema()["properties"].keys())
            elif type(args_schema) == dict:
                valid_args = list(args_schema.keys())
            elif type(args_schema) == list:
                valid_args = args_schema

            # valid_args = list(args_schema.model_json_schema()['properties'].keys())
            incorrect_args = {k: v for k, v in kwargs.items() if k not in valid_args}

            if incorrect_args:
                error_message = "Invalid argument(s) provided: "
                suggestions = []
                for k in incorrect_args:
                    close_matches = difflib.get_close_matches(
                        k, valid_args, n=1, cutoff=0.6
                    )
                    if close_matches:
                        suggestions.append(f"{k}: Did you mean '{close_matches[0]}'?")
                    else:
                        suggestions.append(
                            f"{k}: This argument is not recognized and will be ignored."
                        )
                return error_message + ", ".join(suggestions)
            # Filter the kwargs and args and call the original function
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
            return func(**filtered_kwargs)

        return wrapper

    return decorator


def validate_tool_args(args_schema):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, **kwargs):
            if issubclass(args_schema, BaseModel):
                valid_args = list(args_schema.model_json_schema()["properties"].keys())
            elif type(args_schema) == dict:
                valid_args = list(args_schema.keys())
            elif type(args_schema) == list:
                valid_args = args_schema

            # valid_args = list(args_schema.model_json_schema()['properties'].keys())
            incorrect_args = {k: v for k, v in kwargs.items() if k not in valid_args}
            if incorrect_args:
                error_message = "Invalid argument(s) provided: "
                suggestions = []
                for k in incorrect_args:
                    close_matches = difflib.get_close_matches(
                        k, valid_args, n=1, cutoff=0.6
                    )
                    if close_matches:
                        suggestions.append(f"{k}: Did you mean '{close_matches[0]}'?")
                    else:
                        suggestions.append(
                            f"{k}: This argument is not recognized and will be ignored."
                        )
                return error_message + ", ".join(suggestions)

            # Filter the kwargs and args and call the original function
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
            return func(self, **filtered_kwargs)

        return wrapper

    return decorator
