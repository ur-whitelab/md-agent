import difflib
import functools


def validate_arguments(valid_args):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract the names of the provided keyword arguments
            provided_kwargs = kwargs.keys()
            # Find incorrect arguments that are not in the list of valid arguments
            incorrect_kwargs = [arg for arg in provided_kwargs if arg not in valid_args]
            # Suggest corrections for incorrect arguments
            if incorrect_kwargs:
                corrections = {
                    arg: difflib.get_close_matches(arg, valid_args, n=1, cutoff=0.6)
                    for arg in incorrect_kwargs
                }
                error_message = "Invalid argument(s) provided. "
                suggestions = []
                for incorrect, correct in corrections.items():
                    if correct:
                        suggestions.append(
                            f"'{incorrect}': " f"(maybe you mean: '{correct[0]}'?)\n"
                        )
                    else:
                        suggestions.append(
                            f"'{incorrect}': (This argument is not "
                            "used by the tool, it will "
                            "be ignored)\n"
                        )
                error_message += " ".join(suggestions)
                raise ValueError(error_message)
            if type(args) == dict:
                provided_args = args.keys()
                incorrect_args = [arg for arg in provided_args if arg not in valid_args]
                if incorrect_args:
                    corrections = {
                        arg: difflib.get_close_matches(arg, valid_args, n=1, cutoff=0.6)
                        for arg in incorrect_args
                    }
                    error_message = "Invalid argument(s) provided. "
                    suggestions = []
                    for incorrect, correct in corrections.items():
                        if correct:
                            suggestions.append(
                                f"'{incorrect}': "
                                f"(maybe you mean: '{correct[0]}'?)\n"
                            )
                        else:
                            suggestions.append(
                                f"'{incorrect}': (This argument is not "
                                "used by the tool, it will "
                                "be ignored)\n"
                            )
                    error_message += " ".join(suggestions)
                    raise ValueError(error_message)
            return None

        return wrapper

    return decorator
