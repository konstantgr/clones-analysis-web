import os
import json


def get_results(results_dir):
    results = []
    for file in filter(lambda x: x.endswith(".json"), os.listdir(results_dir)):
        results.append(
            ClonesResult(os.path.join(results_dir, file))
        )
    return results


def pos2code(pos, code):
    start = code[:pos[0][0]].count('\n')
    end = code[:pos[-1][1]].count('\n')
    lines = code.splitlines()

    # s = ''
    # for (start, end) in pos:
    #     s += code[start:end] + " "

    res = "\n".join(lines[start:end]) if start != end else lines[start]
    return res, start, end


def trim_clone(pos, code):
    s = ''
    for i, (start, end) in enumerate(pos):
        if code[:start].splitlines()[0].strip() == '':
            break
        else:
            del pos[i]

    l = len(pos) - 1
    for i, (start, end) in enumerate(pos[::-1]):
        print(code[end:].splitlines()[0].strip())
        if code[end:].splitlines()[0].strip() == '':
            break
        else:
            del pos[l - i]


class ClonesResult:
    def __init__(self, path):
        self.path = path
        self.directory, self.filename = self.get_metadata(path)

        self.owner, self.repository, self.name = self.filename.split("#")
        self.name = self.name.removesuffix('.json')

        self.clones_lengths, self.all_clones, self.tree_length = self.read_clones_data()

    def read_clones_data(self):
        prohibited_keys = {'file_name', 'project_name', 'initial_tree_length'}
        with open(self.path, 'r') as f:
            json_data = json.load(f)

        lengths = [k for k in list(json_data.keys()) if k not in prohibited_keys]
        clones = json.loads(json_data["1"])
        tree_length = json_data['initial_tree_length']

        return list(map(int, lengths)), clones, tree_length

    def get_clones_stats(self, min_length):
        res = []
        for group in self.all_clones['clonesPositions']:
            if len(group[0]) >= min_length:
                res.append(group)
        return res

    def get_clones_count(self, min_length):
        cnt = 0
        print(self.all_clones.keys())
        for group in filter(lambda x: x[0] > min_length, self.all_clones['clonesPositions']):
            cnt += len(group)

        return cnt

    @staticmethod
    def get_metadata(path):
        split_path = path.split('/')
        directory = split_path[:-1]
        filename = split_path[-1]
        return directory, filename

