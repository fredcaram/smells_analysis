import re

def extract_namespace_from_path(full_name, verbose=False):
    m = re.match(r'(.*)[.][A-Z].*', full_name)

    if not m is None:
        namespace = m.group(1)
        return namespace

    if verbose:
        print("Warning: Couldn't extract namespace from path: {0}".format(full_name))
    return ""


def extract_class_from_path(full_name, verbose=False):
    m = re.match(r'.*[.]([A-Z]\w*)[.]?.*', full_name)

    if not m is None:
        class_ = m.group(1)
        return class_

    if verbose:
        print("Warning: Couldn't extract class from path: {0}".format(full_name))
        return ""

def extract_method_from_path(full_name, verbose=False):
    m = re.match(r'.*[.][A-Z]\w*[.:]([a-z]\w*([(].*[)])?)[.]?.*', full_name)

    if not m is None:
        method = m.group(1)
        return method

    if verbose:
        print("Warning: Couldn't extract method from path: {0}".format(full_name))
        return ""


def decompose_class_members(full_name, verbose=False):
    namespace = extract_namespace_from_path(full_name, verbose)
    class_ = extract_class_from_path(full_name, verbose)
    method = extract_method_from_path(full_name, verbose)
    return {"namespace": namespace, "class": class_, "method": method}


def extract_class_from_method(method_desc, verbose=False):
    members = decompose_class_members(method_desc)
    return "{0}.{1}".format(members["namespace"], members["class"])

def extract_path_until_method(method_desc, verbose=False):
    members = decompose_class_members(method_desc)
    method = "{0}.{1}".format(members["namespace"], members["class"])
    if (not members is None) and (members["method"] != ""):
        method = method + ".{0}".format(members["method"])
    return method