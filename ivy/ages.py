"""
Calculate node ages from branch lengths.

The function of interest is `ages2lengths`
"""

def ages2lengths(node, node_ages, results={}):
    "convert node ages to branch lengths"
    for d in node.descendants():
        age = node_ages[d]
        if d.parent:
            parent_age = node_ages[d.parent]
            results[d] = parent_age - age
    return results

def min_ages(node, leaf_ages, results={}):
    "calculate minimum ages given fixed ages in leaf_ages"
    v = []
    for child in node.children:
        if child.label and (child.label in leaf_ages):
            age = leaf_ages[child.label]
            v.append(age)
            results[child] = age
        else:
            min_ages(child, leaf_ages, results)
            age = results[child]
            v.append(age)
    results[node] = max(v)
    return results

def smooth(node, node_ages, results={}):
    """
    adjust ages of internal nodes by smoothing
    """
    if node.parent:
        parent_age = node_ages[node.parent]
        if node.children:
            max_child_age = max([ node_ages[child] for child in node.children ])
            # make the new age the average of parent and max child
            new_node_age = (parent_age + max_child_age)/2.0
            results[node] = new_node_age
        else:
            results[node] = node_ages[node]
    else:
        results[node] = node_ages[node]
    for child in node.children:
        smooth(child, node_ages, results)
    return results

if __name__ == "__main__":
    import newick, ascii

    s = "((((a,b),(c,d),(e,f)),g),h);"
    root = newick.parse(s)

    leaf_ages = {
        "a": 3,
        "b": 2,
        "c": 4,
        "d": 1,
        "e": 3,
        "f": 0.5,
        "g": 10,
        "h": 5,
        }

    ma = min_ages(root, leaf_ages)
    d = ma
    for i in range(10):
        d = smooth(root, d)
    for node, val in ages2lengths(root, d).items():
        node.length = val
    print ascii.render(root, scaled=1)
