import os


def findfiles(target, extension='.csv'):
    """Searches a specified directory and locates all absolute file paths.

        Arguments:
            - target    String path to the target directory to be searched
            - extension String of the format '.jpg' to filter by
        Returns:
            - list of file paths (iterable)
    """
    data = []

    for path, subFolders, files in os.walk(target):
        files.sort()
        files = [file for file in files if file.endswith(extension)]
        for file in files:
            p = os.path.join(os.path.abspath(path), file)
            data.append(p)
            print(p)

    return data
