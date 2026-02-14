import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class Vertex:
    def __init__(self, key, color=None):
        self.key = key
        self.color = color

    def get_color(self):
        return self.color

    def __eq__(self, other) -> bool:
        if isinstance(other, Vertex):
            return self.key == other.key
        return False

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return str(self.key)

class GraphList:
    def __init__(self):
        self.graph = {}

    def is_empty(self) -> bool:
        return len(self.graph) == 0

    def insert_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def insert_edge(self, vertex1, vertex2, edge=1):
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1][vertex2] = edge
            self.graph[vertex2][vertex1] = edge


    def delete_vertex(self, vertex):
        if vertex in self.graph:
            del self.graph[vertex]
            for i in self.graph:
                if vertex in self.graph[i]:
                    del self.graph[i][vertex]

    def delete_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1].pop(vertex2)
            self.graph[vertex2].pop(vertex1)


    def neighbours(self, vertex_id):
        return self.graph[vertex_id].items()

    def neighbours2(self, vertex):
        return self.graph[vertex.key]
    def vertices(self):
        return self.graph.keys()

    def get_vertex(self, vertex_id):
        return vertex_id

    def get_edge(self, vertex1, vertex2):
        return self.graph[vertex1][vertex2]

    def plot_graph(self, v_color, e_color):
        for vertex in self.graph:
            v = vertex
            y, x = v.key
            plt.scatter(x, y, c=v_color)

            for n_idx, _ in self.neighbours(vertex):
                yn, xn = self.get_vertex(n_idx).key
                plt.plot([x, xn], [y, yn], color=e_color)

        # plt.gca().invert_yaxis()
        # plt.show()



    def to_dict(self):
        graph_dict = {}
        for vertex in self.graph:
            neighbors = {str(neighbor.key): edge for neighbor, edge in self.neighbours(vertex)}
            graph_dict[str(vertex.key)] = neighbors
        return graph_dict
def length(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def angle(x1, y1, x2, y2):
    return np.arctan2(np.abs(y2 - y1), np.abs(x2 - x1))

def fill_biometric_graph_from_image(I, graph):
    X, Y = I.shape
    for x in range(X):
        for y in range(Y):
            if I[x, y] == 255:
                v = Vertex((x, y))
                graph.insert_vertex(v)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        x2 = x + i
                        y2 = y + j
                        if 0 <= x2 < X and 0 <= y2 < Y and I[x2, y2] == 255:
                            v2 = Vertex((x2, y2))
                            l = length(x, y, x2, y2)
                            a = angle(x, y, x2, y2)
                            graph.insert_edge(v, v2, (l, a))


def graph_path(graph, start_v, prev_v):
    current = start_v
    path = [prev_v]
    path.append(current)

    while current is not None:
        neighbours = list(graph.neighbours(current))
        if len(neighbours) == 2:
            if neighbours[0][0] != prev_v:
                next_v = neighbours[0][0]
            else:
                next_v = neighbours[1][0]
            prev_v = current
            current = next_v
            path.append(current)
            if len(graph.neighbours(current)) != 2:
                break
        else:
            break

    return path


def unclutter_biometric_graph(graph):
    to_remove = []
    to_add = []

    for v in list(graph.vertices()):
        neighbours = list(graph.neighbours(v))
        if len(neighbours) == 2:
            to_remove.append(v)

        if len(neighbours) != 2:
            for n, e in neighbours:
                path = graph_path(graph, n, v)
                if len(path) > 1:

                    for i in path[1:-1]:
                        to_remove.append(i)
                    last = path[-1]
                    to_add.append((v, last))

    for v in to_remove:
        graph.delete_vertex(v)

    for v1, v2 in to_add:
        l = length(v1.key[0], v1.key[1], v2.key[0], v2.key[1])
        a = angle(v1.key[0], v1.key[1], v2.key[0], v2.key[1])
        graph.insert_vertex(v1)
        graph.insert_vertex(v2)
        graph.insert_edge(v1, v2, (l, a))

def merge_near_vertices(graph, thr):
    to_merge = []
    visited = []
    for v in list(graph.vertices()):

        if v not in visited:
            close_to_v = []
            close_to_v.append(v)
            # neighbours = list(graph.neighbours(v))
            # for n, e in neighbours:
            #     if length(v.key[0], v.key[1], n.key[0], n.key[1]) < thr:
            #         close_to_v.append(n)
            #         visited.append(n)
            # if len(close_to_v) > 1:
            #     to_merge.append(close_to_v)
            # visited.append(v)
            for u in list(graph.vertices()):
                if u != v and u not in visited:
                    if length(v.key[0], v.key[1], u.key[0], u.key[1]) < thr:
                        close_to_v.append(u)
                        visited.append(u)
            if len(close_to_v) > 1:
                to_merge.append(close_to_v)
            visited.append(v)

    new_vertices = []
    for i in to_merge:
        # print(i)
        mean_x = int(np.mean([v.key[0] for v in i]))
        mean_y = int(np.mean([v.key[1] for v in i]))
        # for point in i:
        #     graph.delete_vertex(point)
        v = Vertex((mean_x, mean_y))
        graph.insert_vertex(v)
        new_vertices.append(v)

        for point in i:
            if point in list(graph.vertices()):
                neighbours = list(graph.neighbours(point))
                # print(neighbours)
                for n, e in neighbours:
                    if n not in i:
                        if length(point.key[0], point.key[1], n.key[0], n.key[1]) > thr:
                            l = length(point.key[0], point.key[1], n.key[0], n.key[1])
                            a = angle(point.key[0], point.key[1], n.key[0], n.key[1])
                            graph.insert_edge(v, n, (l, a))

        for point in i:
            if point != v:
                graph.delete_vertex(point)







def biometric_graph_registration(g1, g2, Ni, eps):
    c = 0
    best_t = None
    best_angle = None

    edges1 = []
    for v in list(g1.vertices()):
        for n, e in list(g1.neighbours(v)):
            edges1.append((v, n, e))

    edges2 = []
    for v in list(g2.vertices()):
        for n, e in list(g2.neighbours(v)):
            edges1.append((v, n, e))

    sab_values = []
    for (v1, n1, (l1, a1)) in edges1:
        for (v2, n2, (l2, a2)) in edges2:
            sab = np.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2) / (0.5 * (l1 + l2))
            sab_values.append(((v1, n1), (v2, n2), sab))

    sab_values.sort(key=lambda x: x[2])

    Ni = min(len(sab_values), Ni)
    for (v1, n1), (v2, n2), _ in sab_values[:Ni]:
        t = (-v1.key[0], -v1.key[1])
        angle1 = np.arctan2(n1.key[1] - v1.key[1], n1.key[0] - v1.key[0])
        angle2 = np.arctan2(n2.key[1] - v2.key[1], n2.key[0] - v2.key[0])
        angle_rot = angle2 - angle1

        graph2_rotated = GraphList()
        for vertex in list(g2.vertices()):
            graph2_rotated.insert_vertex(vertex)
        for vertex in list(g2.vertices()):
            for neighbor, edge in g2.neighbours(vertex):
                graph2_rotated.insert_edge(vertex, neighbor, edge)
        rotate_translate_graph(graph2_rotated, t, angle_rot)

        merge_near_vertices(graph2_rotated, eps)

        count = 0
        for vert in list(g1.vertices()):
            if vert in list(graph2_rotated.vertices()):
                # if length(vert1.key[0], vert1.key[1], vert2.key[0], vert2.key[1]):
                count += 1


        similar = count / max(len(list(g1.vertices())), len(list(graph2_rotated.vertices())))

        if similar > c:
            c = similar
        best_t = t
        best_angle = angle_rot

    if best_t is not None and best_angle is not None:
        rotate_translate_graph(g2, best_t, best_angle)
        merge_near_vertices(g2, eps)
    return g1, g2

def rotate_translate_graph(graph, t, a):
    sin_th = np.sin(a)
    cos_th = np.cos(a)
    position = {}
    tx, ty = t
    for vertex in list(graph.vertices()):
        x, y = vertex.key
        x_new = (x + tx)* cos_th - (ty + y) * sin_th
        y_new = (x + tx) * sin_th + (y + ty) * cos_th
        position[vertex] = (x_new, y_new)
        

    for vertex, (x_new, y_new) in position.items():
        vertex.key = (x_new, y_new)

    for v in list(graph.vertices()):
        for n, edge in list(graph.neighbours(v)):
            if n in position:
                l, th = edge
                th_new = th+ a
                graph.insert_edge(v, n, (l, th_new))


if __name__ == '__main__':
    data_path = "./images"
    img_level = "easy"
    img_list = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    input_data = []
    for img_name in img_list:
        if img_name[-3:] == "png":
            if img_name.split('_')[-2] == img_level:
                print("Processing ", img_name, "...")

                img = cv2.imread(os.path.join(data_path, img_name))
                img_1ch = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img_bin = cv2.threshold(img_1ch, 127, 255, cv2.THRESH_BINARY)

                graph = GraphList()
                fill_biometric_graph_from_image(img_bin, graph)
                # graph.plot_graph('r', 'g')
                # graph_dict = graph.to_dict()
                # print(graph_dict)
                unclutter_biometric_graph(graph)
                merge_near_vertices(graph, thr=5)
                input_data.append((img_name, graph))
                print("Saved!")

    for i in range(len(input_data)):
        for j in range(len(input_data)):
            graph1_input = input_data[i][1]
            graph2_input = input_data[j][1]

            graph1, graph2 = biometric_graph_registration(graph1_input, graph2_input, Ni=50, eps=10)


            plt.figure()
            graph1.plot_graph(v_color='red', e_color='green')
            graph2.plot_graph(v_color='gold', e_color='blue')
            plt.title('Graph comparison')

            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()