import re


def extract_class_from_method(method_desc):
    m = re.match("(.*)[.]\w+\(.*\)", method_desc)
    if m is None:
        print("Warning: Couldn't extract class from method: {0}".format(method_desc))
        return method_desc
    class_ = m.group(1)
    return class_