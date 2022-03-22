import trt_pose.plugins
import json


class ParseObjects(object):
    
    def __init__(self, topology, cmap_threshold=0.1, link_threshold=0.1, cmap_window=5, line_integral_samples=7, max_num_parts=100, max_num_objects=100):
        self.topology = topology
        self.cmap_threshold = cmap_threshold
        self.link_threshold = link_threshold
        self.cmap_window = cmap_window
        self.line_integral_samples = line_integral_samples
        self.max_num_parts = max_num_parts
        self.max_num_objects = max_num_objects
    
    def __call__(self, cmap, paf):
        peak_counts, peaks = trt_pose.plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = trt_pose.plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        score_graph = trt_pose.plugins.paf_score_graph(paf, self.topology, peak_counts, normalized_peaks, self.line_integral_samples)
        connections = trt_pose.plugins.assignment(score_graph, self.topology, peak_counts, self.link_threshold)
        object_counts, objects =trt_pose.plugins.connect_parts(connections, self.topology, peak_counts, self.max_num_objects)
        
        return object_counts, objects, normalized_peaks


class PostProcess(object):
    def __init__(self, task, *args, **kwargs):
        super().__init__()
        if isinstance(task, str):
            with open(task, 'r') as f:
                task = json.load(f)
        self.task = task
        self.topology = trt_pose.coco.coco_category_to_topology(self.task)
        self.parse_objects = ParseObjects(self.topology, *args, **kwargs)

    def __call__(self, cmap, paf):
        object_counts, objects, normalized_peaks = self.parse_objects(cmap, paf)

        topology = self.topology
        K = topology.shape[0]
        count = int(object_counts[0])
        objects_py = []
        for i in range(count):
            obj_py = {}
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = float(peak[1])
                    y = float(peak[0])
                    label = self.task['keypoints'][j]
                    obj_py[label] = {
                        'x': x,
                        'y': y
                    }
            objects_py.append(obj_py)
        return objects_py