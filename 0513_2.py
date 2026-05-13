# -*- coding: utf-8 -*-
"""
Fluent数据四边形区域均值计算工具（倾斜面可指定方向轴，矩阵排序自适应）
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon
import os
from scipy.spatial import ConvexHull
from scipy.linalg import svd
from scipy.optimize import least_squares
from scipy.spatial import KDTree
import warnings
warnings.filterwarnings('ignore')

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GeometryAnalyzer:
    def __init__(self, df, coord_cols, quad_type, const_axis=None, const_value=None,
                 length_axis=None, gravity_axis=None):
        self.df = df
        self.coord_cols = coord_cols
        self.quad_type = quad_type
        self.const_axis = const_axis
        self.const_value = const_value
        self.length_axis = length_axis
        self.gravity_axis = gravity_axis
        self.geo_params = {}
        self.vertices = None
        self._process_data()

    def _process_data(self):
        if self.quad_type == 'plane':
            self._filter_by_const_axis()
        else:
            self._fit_plane_and_project()

    def _filter_by_const_axis(self):
        if self.const_axis == 'X':
            self.df_plane = self.df[np.abs(self.df[self.coord_cols[0]] - self.const_value) <= 1e-6].copy()
            self.x_col = self.coord_cols[1]
            self.y_col = self.coord_cols[2]
            self.plane_x_label = 'Y'
            self.plane_y_label = 'Z'
            self.plane_axes = [self.coord_cols[1], self.coord_cols[2]]
        elif self.const_axis == 'Y':
            self.df_plane = self.df[np.abs(self.df[self.coord_cols[1]] - self.const_value) <= 1e-6].copy()
            self.x_col = self.coord_cols[0]
            self.y_col = self.coord_cols[2]
            self.plane_x_label = 'X'
            self.plane_y_label = 'Z'
            self.plane_axes = [self.coord_cols[0], self.coord_cols[2]]
        else:
            self.df_plane = self.df[np.abs(self.df[self.coord_cols[2]] - self.const_value) <= 1e-6].copy()
            self.x_col = self.coord_cols[0]
            self.y_col = self.coord_cols[1]
            self.plane_x_label = 'X'
            self.plane_y_label = 'Y'
            self.plane_axes = [self.coord_cols[0], self.coord_cols[1]]

        if len(self.df_plane) == 0:
            raise ValueError(f"未找到{self.const_axis}={self.const_value}平面上的数据点")

        self.coord1_vals = self.df_plane[self.x_col].values
        self.coord2_vals = self.df_plane[self.y_col].values
        self.plane_label = f"{self.const_axis} = {self.const_value}"
        self._compute_quadrilateral_params()

    def _fit_plane_and_project(self):
        """正交投影模式：直接使用原始X/Y/Z坐标方向进行投影"""
        len_axis = self.length_axis  # 用户选择的管长方向（X/Y/Z）
        grav_axis = self.gravity_axis  # 用户选择的重力方向（X/Y/Z）
        
        # 坐标列映射：确保使用正确的列名
        coord_map = {'X': self.coord_cols[0], 'Y': self.coord_cols[1], 'Z': self.coord_cols[2]}
        
        # 设置平面内的坐标轴：管长方向作为U轴（x_col），重力方向作为V轴（y_col）
        self.x_col = coord_map[len_axis]
        self.y_col = coord_map[grav_axis]
        self.plane_x_label = len_axis  # 直接使用用户选择的轴标签
        self.plane_y_label = grav_axis  # 直接使用用户选择的轴标签
        self.plane_axes = [self.x_col, self.y_col]
        
        # 正交投影：直接使用原始坐标值，不做任何变换
        # 相当于丢弃第三个坐标，保留管长和重力方向的坐标
        self.df_plane = self.df.copy()
        self.coord1_vals = self.df_plane[self.x_col].values
        self.coord2_vals = self.df_plane[self.y_col].values
        
        # 显示投影平面信息
        drop_axis = next(ax for ax in ['X', 'Y', 'Z'] if ax not in [len_axis, grav_axis])
        self.plane_label = f"正交投影 (丢弃{drop_axis}坐标，保留{len_axis}×{grav_axis})"

        self._compute_quadrilateral_params()

    def _compute_quadrilateral_params(self):
        points_2d = np.column_stack((self.coord1_vals, self.coord2_vals))

        try:
            hull = ConvexHull(points_2d)
            boundary = points_2d[hull.vertices]
        except:
            boundary = np.array([[np.min(self.coord1_vals), np.min(self.coord2_vals)],
                                 [np.max(self.coord1_vals), np.min(self.coord2_vals)],
                                 [np.max(self.coord1_vals), np.max(self.coord2_vals)],
                                 [np.min(self.coord1_vals), np.max(self.coord2_vals)]])

        n = len(boundary)
        edges = []
        for i in range(n):
            p1 = boundary[i]
            p2 = boundary[(i+1) % n]
            length = np.linalg.norm(p2 - p1)
            edges.append((length, p1, p2))

        edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
        longest_edges = edges_sorted[:4]

        vertex_set = set()
        for _, p1, p2 in longest_edges:
            vertex_set.add(tuple(p1))
            vertex_set.add(tuple(p2))
        vertices = [np.array(v) for v in vertex_set]

        if len(vertices) != 4:
            def line_intersection_extend(p1, p2, q1, q2):
                A1 = p2[1] - p1[1]
                B1 = p1[0] - p2[0]
                C1 = A1 * p1[0] + B1 * p1[1]
                A2 = q2[1] - q1[1]
                B2 = q1[0] - q2[0]
                C2 = A2 * q1[0] + B2 * q1[1]
                det = A1 * B2 - A2 * B1
                if abs(det) < 1e-10:
                    return None
                x = (B2 * C1 - B1 * C2) / det
                y = (A1 * C2 - A2 * C1) / det
                return np.array([x, y])

            midpoints = [(p1 + p2) / 2 for _, p1, p2 in longest_edges]
            center = np.mean(midpoints, axis=0)
            angles = [np.arctan2(m[1]-center[1], m[0]-center[0]) for m in midpoints]
            sorted_indices = np.argsort(angles)
            sorted_edges = [longest_edges[i] for i in sorted_indices]

            new_vertices = []
            for k in range(4):
                _, p1, p2 = sorted_edges[k]
                _, q1, q2 = sorted_edges[(k+1) % 4]
                inter = line_intersection_extend(p1, p2, q1, q2)
                if inter is not None:
                    new_vertices.append(inter)
                else:
                    mid1 = (p1 + p2) / 2
                    mid2 = (q1 + q2) / 2
                    new_vertices.append((mid1 + mid2) / 2)

            unique_vertices = []
            for v in new_vertices:
                if not any(np.linalg.norm(v - uv) < 1e-6 for uv in unique_vertices):
                    unique_vertices.append(v)
            if len(unique_vertices) == 4:
                vertices = unique_vertices
            else:
                min1, max1 = np.min(self.coord1_vals), np.max(self.coord1_vals)
                min2, max2 = np.min(self.coord2_vals), np.max(self.coord2_vals)
                vertices = [np.array([min1, min2]), np.array([max1, min2]),
                            np.array([max1, max2]), np.array([min1, max2])]

        if len(vertices) == 4:
            hull_pts = ConvexHull(vertices).vertices
            self.vertices = [vertices[i] for i in hull_pts]
            side_lengths = []
            for i in range(4):
                p1 = self.vertices[i]
                p2 = self.vertices[(i+1) % 4]
                side_lengths.append(np.linalg.norm(p2 - p1))
            self.geo_params['fit_side_lengths'] = side_lengths
            self.geo_params['fit_perimeter'] = sum(side_lengths)
        else:
            min1, max1 = np.min(self.coord1_vals), np.max(self.coord1_vals)
            min2, max2 = np.min(self.coord2_vals), np.max(self.coord2_vals)
            self.vertices = [np.array([min1, min2]), np.array([max1, min2]),
                             np.array([max1, max2]), np.array([min1, max2])]
            side_lengths = [max1-min1, max2-min2, max1-min1, max2-min2]
            self.geo_params['fit_side_lengths'] = side_lengths
            self.geo_params['fit_perimeter'] = 2 * ((max1-min1)+(max2-min2))

        self.geo_params['coord1_min'] = np.min(self.coord1_vals)
        self.geo_params['coord1_max'] = np.max(self.coord1_vals)
        self.geo_params['coord2_min'] = np.min(self.coord2_vals)
        self.geo_params['coord2_max'] = np.max(self.coord2_vals)
        self.geo_params['length'] = max(self.geo_params['coord1_max'] - self.geo_params['coord1_min'],
                                        self.geo_params['coord2_max'] - self.geo_params['coord2_min'])
        self.geo_params['width'] = min(self.geo_params['coord1_max'] - self.geo_params['coord1_min'],
                                       self.geo_params['coord2_max'] - self.geo_params['coord2_min'])
        self.geo_params['perimeter'] = 2 * (self.geo_params['length'] + self.geo_params['width'])
        self.geo_params['quad_type'] = self.quad_type

        if self.quad_type == 'plane':
            if self.const_axis == 'X':
                self.geo_params['x_min'] = self.const_value
                self.geo_params['x_max'] = self.const_value
                self.geo_params['y_min'] = self.geo_params['coord1_min']
                self.geo_params['y_max'] = self.geo_params['coord1_max']
                self.geo_params['z_min'] = self.geo_params['coord2_min']
                self.geo_params['z_max'] = self.geo_params['coord2_max']
            elif self.const_axis == 'Y':
                self.geo_params['x_min'] = self.geo_params['coord1_min']
                self.geo_params['x_max'] = self.geo_params['coord1_max']
                self.geo_params['y_min'] = self.const_value
                self.geo_params['y_max'] = self.const_value
                self.geo_params['z_min'] = self.geo_params['coord2_min']
                self.geo_params['z_max'] = self.geo_params['coord2_max']
            else:
                self.geo_params['x_min'] = self.geo_params['coord1_min']
                self.geo_params['x_max'] = self.geo_params['coord1_max']
                self.geo_params['y_min'] = self.geo_params['coord2_min']
                self.geo_params['y_max'] = self.geo_params['coord2_max']
                self.geo_params['z_min'] = self.const_value
                self.geo_params['z_max'] = self.const_value
        else:
            self.geo_params['x_min'] = None
            self.geo_params['x_max'] = None
            self.geo_params['y_min'] = None
            self.geo_params['y_max'] = None
            self.geo_params['z_min'] = None
            self.geo_params['z_max'] = None

    def get_boundary_points(self):
        points_2d = np.column_stack((self.coord1_vals, self.coord2_vals))
        try:
            hull = ConvexHull(points_2d)
            boundary = points_2d[hull.vertices]
        except:
            boundary = np.array([[np.min(self.coord1_vals), np.min(self.coord2_vals)],
                                 [np.max(self.coord1_vals), np.min(self.coord2_vals)],
                                 [np.max(self.coord1_vals), np.max(self.coord2_vals)],
                                 [np.min(self.coord1_vals), np.max(self.coord2_vals)]])
        center = np.mean(boundary, axis=0)
        angles = np.arctan2(boundary[:,1]-center[1], boundary[:,0]-center[0])
        return boundary[np.argsort(angles)]

    def get_plane_points(self):
        if self.quad_type == 'plane':
            return self.df_plane, self.coord1_vals, self.coord2_vals
        else:
            df_copy = self.df_plane.copy()
            df_copy['U_proj'] = self.coord1_vals
            df_copy['V_proj'] = self.coord2_vals
            return df_copy, self.coord1_vals, self.coord2_vals

    def compute_natural_coordinates(self, x, y):
        p0, p1, p2, p3 = self.vertices
        def residuals(uv):
            u, v = uv
            px = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
            py = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
            return np.array([px - x, py - y])
        res = least_squares(residuals, [0.5, 0.5], bounds=([0,0],[1,1]))
        return res.x[0], res.x[1]


class FluentDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.coord_cols = ['X', 'Y', 'Z']
        self.value_col = None
        self.area_col = 'cell-surface-area'
        self.geometry = None
        self.geo_params = {}
        self.region_info = None
        self.df_plane_labeled = None
        self.length_axis = None
        self.gravity_axis = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_excel(self.filepath)
            possible_x = [c for c in self.df.columns if c.upper() in ['X', 'COORD_X']]
            possible_y = [c for c in self.df.columns if c.upper() in ['Y', 'COORD_Y']]
            possible_z = [c for c in self.df.columns if c.upper() in ['Z', 'COORD_Z']]
            if len(possible_x) > 0:
                self.coord_cols[0] = possible_x[0]
            if len(possible_y) > 0:
                self.coord_cols[1] = possible_y[0]
            if len(possible_z) > 0:
                self.coord_cols[2] = possible_z[0]
            if self.area_col not in self.df.columns:
                raise ValueError(f"Excel文件中缺少必要的列：'{self.area_col}'")
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col not in self.coord_cols and col != self.area_col:
                    self.value_col = col
                    break
            if self.value_col is None:
                raise ValueError("未找到合适的数值列")
            return True
        except Exception as e:
            messagebox.showerror("错误", f"读取文件失败：{str(e)}")
            return False

    def analyze_quadrilateral(self, quad_type, const_axis=None, const_value=None,
                              length_axis=None, gravity_axis=None):
        self.geometry = GeometryAnalyzer(self.df, self.coord_cols, quad_type,
                                         const_axis, const_value,
                                         length_axis, gravity_axis)
        self.geo_params = self.geometry.geo_params
        self.quad_type = quad_type
        self.length_axis = length_axis
        self.gravity_axis = gravity_axis
        return self.geo_params

    def divide_grid(self, u_divisions, v_divisions, swap_uv=False):
        df_plane, _, _ = self.geometry.get_plane_points()
        if self.quad_type == 'tilted':
            x_col = 'U_proj'
            y_col = 'V_proj'
            df_plane = df_plane.copy()
        else:
            x_col = self.geometry.x_col
            y_col = self.geometry.y_col

        u_vals = np.zeros(len(df_plane))
        v_vals = np.zeros(len(df_plane))
        for i, (xx, yy) in enumerate(zip(df_plane[x_col], df_plane[y_col])):
            u, v = self.geometry.compute_natural_coordinates(xx, yy)
            u_vals[i] = u
            v_vals[i] = v
        df_plane['u_coord'] = u_vals
        df_plane['v_coord'] = v_vals

        if not swap_uv:
            u_divs = u_divisions
            v_divs = v_divisions
        else:
            u_divs = v_divisions
            v_divs = u_divisions

        u_centers = (np.arange(u_divs) + 0.5) / u_divs
        v_centers = (np.arange(v_divs) + 0.5) / v_divs
        centers = []
        for i, uc in enumerate(u_centers):
            for j, vc in enumerate(v_centers):
                if not swap_uv:
                    u_idx = i + 1
                    v_idx = j + 1
                else:
                    u_idx = j + 1
                    v_idx = i + 1
                centers.append((uc, vc, u_idx, v_idx))

        center_points = np.array([(c[0], c[1]) for c in centers])
        tree = KDTree(center_points)
        points = np.column_stack((u_vals, v_vals))
        distances, indices = tree.query(points)

        region_info = []
        df_plane['region_label'] = ''
        for idx, (uc, vc, u_idx, v_idx) in enumerate(centers):
            mask = (indices == idx)
            label = f"区域({u_idx},{v_idx})"
            df_plane.loc[mask, 'region_label'] = label
            region_info.append({
                '区域名称': label,
                'u索引': u_idx,
                'v索引': v_idx,
                'u范围': f"[{uc-0.5/u_divs:.3f}, {uc+0.5/u_divs:.3f}]" if len(u_centers)>0 else "",
                'v范围': f"[{vc-0.5/v_divs:.3f}, {vc+0.5/v_divs:.3f}]"
            })

        self.df_plane_labeled = df_plane
        self.u_edges = np.linspace(0, 1, u_divs + 1)
        self.v_edges = np.linspace(0, 1, v_divs + 1)
        self.plot_x_col = x_col
        self.plot_y_col = y_col
        self.u_divisions = u_divisions
        self.v_divisions = v_divisions
        self.swap_uv = swap_uv
        self.region_info = region_info
        return region_info

    def compute_mean(self, region_info):
        if self.area_col not in self.df_plane_labeled.columns:
            raise ValueError(f"缺少面积列 '{self.area_col}'")
        def weighted_mean(group):
            total_area = group[self.area_col].sum()
            if total_area == 0:
                return 0
            return (group[self.value_col] * group[self.area_col]).sum() / total_area
        grouped = self.df_plane_labeled.groupby('region_label').apply(weighted_mean).reset_index()
        grouped.columns = ['region_label', f'{self.value_col}_加权均值']
        counts = self.df_plane_labeled.groupby('region_label').size().reset_index(name='点数')
        grouped = pd.merge(grouped, counts, on='region_label')
        region_df = pd.DataFrame(region_info)
        result = pd.merge(region_df, grouped, left_on='区域名称', right_on='region_label', how='left')
        result.fillna(0, inplace=True)
        if 'region_label' in result.columns:
            result.drop('region_label', axis=1, inplace=True)
        return result

    def export_result(self, result_df, output_path):
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            geo_df = pd.DataFrame([self.geo_params])
            geo_df.to_excel(writer, sheet_name='几何参数', index=False)
            result_df.to_excel(writer, sheet_name='区域均值', index=False)
            export_df = self.df_plane_labeled.copy()
            export_df.to_excel(writer, sheet_name='带标签数据', index=False)
        return output_path


class DivisionVisualizationWindow(tk.Toplevel):
    def __init__(self, parent, processor, region_info, invert_x=False, invert_y=False):
        super().__init__(parent)
        self.title("四边形网格分区可视化")
        self.geometry("800x700")
        self.processor = processor
        self.region_info = region_info
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.create_visualization()

    def create_visualization(self):
        fig, ax = plt.subplots(figsize=(8, 7))

        if self.processor.quad_type == 'tilted':
            df_plane = self.processor.df_plane_labeled
            x = df_plane['U_proj'].values
            y = df_plane['V_proj'].values
            x_label = self.processor.length_axis + ' (投影)'
            y_label = self.processor.gravity_axis + ' (投影)'
            vertices = self.processor.geometry.vertices
        else:
            df_plane = self.processor.df_plane_labeled
            x_axis = self.processor.length_axis
            y_axis = self.processor.gravity_axis
            x = df_plane[x_axis].values
            y = df_plane[y_axis].values
            x_label = x_axis
            y_label = y_axis
            verts = self.processor.geometry.vertices
            source_axis1 = self.processor.geometry.x_col
            source_axis2 = self.processor.geometry.y_col
            verts_user = []
            for v in verts:
                val1 = v[0]
                val2 = v[1]
                x_val = val1 if source_axis1 == x_axis else val2 if source_axis2 == x_axis else None
                y_val = val2 if source_axis2 == y_axis else val1 if source_axis1 == y_axis else None
                if x_val is None or y_val is None:
                    x_val, y_val = v[0], v[1]
                verts_user.append(np.array([x_val, y_val]))
            vertices = verts_user

        ax.scatter(x, y, c='blue', s=10, alpha=0.3)
        poly_pts = np.vstack([vertices, vertices[0]])
        ax.plot(poly_pts[:,0], poly_pts[:,1], 'b-', linewidth=2, label='四边形边界')

        u_edges = self.processor.u_edges
        v_edges = self.processor.v_edges
        u_div = len(u_edges)-1
        v_div = len(v_edges)-1
        swap = self.processor.swap_uv

        for u in u_edges:
            pts = []
            for v in np.linspace(0, 1, 50):
                p0, p1, p2, p3 = vertices
                xc = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
                yc = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
                pts.append([xc, yc])
            pts = np.array(pts)
            ax.plot(pts[:,0], pts[:,1], 'r--', linewidth=1.5, alpha=0.7)
        for v in v_edges:
            pts = []
            for u in np.linspace(0, 1, 50):
                p0, p1, p2, p3 = vertices
                xc = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
                yc = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
                pts.append([xc, yc])
            pts = np.array(pts)
            ax.plot(pts[:,0], pts[:,1], 'r--', linewidth=1.5, alpha=0.7)

        for info in self.region_info:
            u_idx = info['u索引']
            v_idx = info['v索引']
            if not swap:
                uc = (u_edges[u_idx-1] + u_edges[u_idx]) / 2 if u_idx <= u_div else 0.5
                vc = (v_edges[v_idx-1] + v_edges[v_idx]) / 2 if v_idx <= v_div else 0.5
            else:
                uc = (u_edges[v_idx-1] + u_edges[v_idx]) / 2 if v_idx <= u_div else 0.5
                vc = (v_edges[u_idx-1] + v_edges[u_idx]) / 2 if u_idx <= v_div else 0.5
            p0, p1, p2, p3 = vertices
            xc = (1-uc)*(1-vc)*p0[0] + uc*(1-vc)*p1[0] + uc*vc*p2[0] + (1-uc)*vc*p3[0]
            yc = (1-uc)*(1-vc)*p0[1] + uc*(1-vc)*p1[1] + uc*vc*p2[1] + (1-uc)*vc*p3[1]
            ax.text(xc, yc, f"{u_idx},{v_idx}", ha='center', va='center',
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

        p0, p1, p2, p3 = vertices
        if not swap:
            for i, u in enumerate(u_edges):
                u_label = i  # 显示值减1
                if u_label == 0:
                    continue  # 不显示 u=0
                xc = (1-u)*p0[0] + u*p1[0]
                yc = (1-u)*p0[1] + u*p1[1]
                dx = (p1[1]-p0[1]) if abs(p1[1]-p0[1])>0 else 0.05
                dy = (p1[0]-p0[0]) if abs(p1[0]-p0[0])>0 else 0.05
                offset_x = -dy * 0.05
                offset_y = dx * 0.05
                ax.text(xc+offset_x, yc+offset_y, f'U={u_label}', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
            for j, v in enumerate(v_edges):
                v_label = j  # 显示值减1
                if v_label == 0:
                    continue  # 不显示 v=0
                xc = (1-v)*p0[0] + v*p3[0]
                yc = (1-v)*p0[1] + v*p3[1]
                dx = (p3[1]-p0[1]) if abs(p3[1]-p0[1])>0 else 0.05
                dy = (p3[0]-p0[0]) if abs(p3[0]-p0[0])>0 else 0.05
                offset_x = -dy * 0.05
                offset_y = dx * 0.05
                ax.text(xc+offset_x, yc+offset_y, f'V={v_label}', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        else:
            for i, u in enumerate(u_edges):
                v_label = i  # 显示值减1
                if v_label == 0:
                    continue  # 不显示 v=0
                xc = (1-u)*p0[0] + u*p1[0]
                yc = (1-u)*p0[1] + u*p1[1]
                dx = (p1[1]-p0[1]) if abs(p1[1]-p0[1])>0 else 0.05
                dy = (p1[0]-p0[0]) if abs(p1[0]-p0[0])>0 else 0.05
                offset_x = -dy * 0.05
                offset_y = dx * 0.05
                ax.text(xc+offset_x, yc+offset_y, f'V={v_label}', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
            for j, v in enumerate(v_edges):
                u_label = j  # 显示值减1
                if u_label == 0:
                    continue  # 不显示 u=0
                xc = (1-v)*p0[0] + v*p3[0]
                yc = (1-v)*p0[1] + v*p3[1]
                dx = (p3[1]-p0[1]) if abs(p3[1]-p0[1])>0 else 0.05
                dy = (p3[0]-p0[0]) if abs(p3[0]-p0[0])>0 else 0.05
                offset_x = -dy * 0.05
                offset_y = dx * 0.05
                ax.text(xc+offset_x, yc+offset_y, f'U={u_label}', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))

        if self.invert_x:
            ax.invert_xaxis()
        if self.invert_y:
            ax.invert_yaxis()

        ax.set_title(f'四边形网格分区 ({self.processor.u_divisions}×{self.processor.v_divisions})', fontsize=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="关闭", command=self.destroy, bg="#e0e0e0").pack()


class MatrixResultWindow(tk.Toplevel):
    def __init__(self, parent, pivot_df):
        super().__init__(parent)
        self.title("结果矩阵【列=管长方向（u索引），行=重力方向（v索引）】")
        self.geometry("800x600")
        self.pivot_df = pivot_df
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        scroll_y = tk.Scrollbar(frame)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree = ttk.Treeview(frame, yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.config(command=self.tree.yview)
        scroll_x.config(command=self.tree.xview)

        columns = [self.pivot_df.index.name] + list(self.pivot_df.columns)
        self.tree['columns'] = columns
        self.tree['show'] = 'headings'
        self.tree.heading(columns[0], text=columns[0])
        self.tree.column(columns[0], width=100)
        for col in columns[1:]:
            self.tree.heading(col, text=str(col))
            self.tree.column(col, width=100)

        for idx, row in self.pivot_df.iterrows():
            values = [idx] + [row[c] for c in self.pivot_df.columns]
            self.tree.insert('', 'end', values=values)

        self.tree.pack(fill='both', expand=True)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="导出到Excel", command=self.export_to_excel, bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="关闭", command=self.destroy, bg="#e0e0e0").pack(side=tk.LEFT, padx=5)

    def export_to_excel(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if filepath:
            self.pivot_df.to_excel(filepath)
            messagebox.showinfo("完成", f"已导出至：{filepath}")


class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fluent面数据均值工具")
        self.geometry("1050x980")
        self.processor = None
        self.region_info = None
        self.result_df = None
        self.fig = None
        self.division_fig = None
        self.invert_x = False
        self.invert_y = False
        self.swap_uv = False
        self.length_axis_buttons = []
        self.gravity_axis_buttons = []
        self.length_axis_frame = None
        self.gravity_axis_frame = None
        self.create_widgets()

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True, padx=5, pady=5)

        frame_geo = ttk.Frame(notebook)
        notebook.add(frame_geo, text="1. 导入与形状识别")
        self.setup_geo_page(frame_geo)

        frame_divide = ttk.Frame(notebook)
        notebook.add(frame_divide, text="2. 分区与均值计算")
        self.setup_divide_page(frame_divide)

        frame_result = ttk.Frame(notebook)
        notebook.add(frame_result, text="3. 结果与导出")
        self.setup_result_page(frame_result)

        frame_plot = ttk.Frame(notebook)
        notebook.add(frame_plot, text="4. 云图对比")
        self.setup_plot_page(frame_plot)

    # ------------------ 页面1 ------------------
    def setup_geo_page(self, parent):
        row = 0
        tk.Label(parent, text="Excel文件路径：").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.file_path_var = tk.StringVar()
        tk.Entry(parent, textvariable=self.file_path_var, width=55).grid(row=row, column=1, padx=5)
        tk.Button(parent, text="浏览", command=self.select_file, bg="#e0e0e0").grid(row=row, column=2, padx=5)

        row += 1
        remark_label = tk.Label(parent, text="备注：EXCEL表格内需包含坐标列（X,Y,Z）、数值列和面积列'cell-surface-area'",
                                fg="gray", font=('微软雅黑', 8))
        remark_label.grid(row=row, column=0, columnspan=5, sticky='w', padx=5, pady=2)

        row += 1
        tk.Label(parent, text="四边形类型：").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.quad_type_var = tk.StringVar(value='plane')
        self.quad_type_var.trace('w', self.on_quad_type_changed)
        tk.Radiobutton(parent, text="等值面", variable=self.quad_type_var, value='plane').grid(row=row, column=1, sticky='w')
        tk.Radiobutton(parent, text="倾斜面", variable=self.quad_type_var, value='tilted').grid(row=row, column=2, sticky='w')

        self.plane_frame = tk.Frame(parent)
        self.plane_frame.grid(row=row+1, column=0, columnspan=5, sticky='w', padx=5, pady=5)

        tk.Label(self.plane_frame, text="等值坐标轴：").pack(side=tk.LEFT, padx=5)
        self.const_axis_var = tk.StringVar(value='Z')
        self.const_axis_var.trace('w', self.on_const_axis_changed)
        tk.Radiobutton(self.plane_frame, text="X", variable=self.const_axis_var, value='X').pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.plane_frame, text="Y", variable=self.const_axis_var, value='Y').pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(self.plane_frame, text="Z", variable=self.const_axis_var, value='Z').pack(side=tk.LEFT, padx=2)

        tk.Label(self.plane_frame, text="等值坐标值：").pack(side=tk.LEFT, padx=5)
        self.const_value_var = tk.StringVar(value='0')
        tk.Entry(self.plane_frame, textvariable=self.const_value_var, width=10).pack(side=tk.LEFT)

        row += 2
        # 管长方向
        tk.Label(parent, text="管长方向：").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.length_axis_frame = tk.Frame(parent)
        self.length_axis_frame.grid(row=row, column=1, sticky='w')
        self.length_axis_var = tk.StringVar(value='X')
        self.invert_x_btn = tk.Button(parent, text="方向调转", command=self.toggle_invert_x, bg="#e0e0e0")
        self.invert_x_btn.grid(row=row, column=2, padx=5)
        self.invert_x_label = tk.Label(parent, text="正向", fg="green", width=6)
        self.invert_x_label.grid(row=row, column=3, sticky='w')

        row += 1
        # 重力方向
        tk.Label(parent, text="重力方向：").grid(row=row, column=0, sticky='w', padx=5, pady=5)
        self.gravity_axis_frame = tk.Frame(parent)
        self.gravity_axis_frame.grid(row=row, column=1, sticky='w')
        self.gravity_axis_var = tk.StringVar(value='Y')
        self.invert_y_btn = tk.Button(parent, text="方向调转", command=self.toggle_invert_y, bg="#e0e0e0")
        self.invert_y_btn.grid(row=row, column=2, padx=5)
        self.invert_y_label = tk.Label(parent, text="正向", fg="green", width=6)
        self.invert_y_label.grid(row=row, column=3, sticky='w')

        self.update_direction_options()

        row += 1
        self.identify_btn = tk.Button(parent, text="识别四边形并计算几何参数", command=self.identify_geometry, bg="#e0e0e0")
        self.identify_btn.grid(row=row, column=1, pady=10)

        row += 1
        self.geo_info_text = tk.Text(parent, height=8, width=80, font=('Microsoft YaHei', 9))
        self.geo_info_text.grid(row=row, column=0, columnspan=5, padx=5, pady=10)

        row += 1
        self.visualization_frame = tk.Frame(parent, height=350, width=650)
        self.visualization_frame.grid(row=row, column=0, columnspan=5, padx=5, pady=10)
        self.visualization_frame.grid_propagate(False)

        self.on_quad_type_changed()

    def toggle_invert_x(self):
        self.invert_x = not self.invert_x
        self.invert_x_label.config(text="反向" if self.invert_x else "正向", fg="red" if self.invert_x else "green")
        self.preview_division()
        self.visualize_shape()
        # 不再自动调用 compute_regions()，仅刷新预览

    def toggle_invert_y(self):
        self.invert_y = not self.invert_y
        self.invert_y_label.config(text="反向" if self.invert_y else "正向", fg="red" if self.invert_y else "green")
        self.preview_division()
        self.visualize_shape()
        # 不再自动调用 compute_regions()

    def on_quad_type_changed(self, *args):
        self.update_direction_options()
        if self.quad_type_var.get() == 'plane':
            self.plane_frame.grid()
        else:
            self.plane_frame.grid_remove()

    def on_const_axis_changed(self, *args):
        if self.quad_type_var.get() == 'plane':
            self.update_direction_options()

    def update_direction_options(self):
        for btn in self.length_axis_buttons:
            btn.destroy()
        for btn in self.gravity_axis_buttons:
            btn.destroy()
        self.length_axis_buttons.clear()
        self.gravity_axis_buttons.clear()

        if self.quad_type_var.get() == 'plane':
            if self.processor is not None:
                const_axis = self.const_axis_var.get()
                if const_axis == 'X':
                    plane_axes = [self.processor.coord_cols[1], self.processor.coord_cols[2]]
                elif const_axis == 'Y':
                    plane_axes = [self.processor.coord_cols[0], self.processor.coord_cols[2]]
                else:
                    plane_axes = [self.processor.coord_cols[0], self.processor.coord_cols[1]]
            else:
                const_axis = self.const_axis_var.get()
                if const_axis == 'X':
                    plane_axes = ['Y', 'Z']
                elif const_axis == 'Y':
                    plane_axes = ['X', 'Z']
                else:
                    plane_axes = ['X', 'Y']
            for ax in plane_axes:
                rb = tk.Radiobutton(self.length_axis_frame, text=f"{ax}轴", variable=self.length_axis_var, value=ax)
                rb.pack(side=tk.LEFT, padx=5)
                self.length_axis_buttons.append(rb)
            for ax in plane_axes:
                rb = tk.Radiobutton(self.gravity_axis_frame, text=f"{ax}轴", variable=self.gravity_axis_var, value=ax)
                rb.pack(side=tk.LEFT, padx=5)
                self.gravity_axis_buttons.append(rb)
            if self.length_axis_var.get() not in plane_axes:
                self.length_axis_var.set(plane_axes[0])
            if self.gravity_axis_var.get() not in plane_axes:
                self.gravity_axis_var.set(plane_axes[1] if len(plane_axes)>1 else plane_axes[0])
        else:
            all_axes = ['X', 'Y', 'Z']
            for ax in all_axes:
                rb = tk.Radiobutton(self.length_axis_frame, text=f"{ax}轴", variable=self.length_axis_var, value=ax)
                rb.pack(side=tk.LEFT, padx=5)
                self.length_axis_buttons.append(rb)
            for ax in all_axes:
                rb = tk.Radiobutton(self.gravity_axis_frame, text=f"{ax}轴", variable=self.gravity_axis_var, value=ax)
                rb.pack(side=tk.LEFT, padx=5)
                self.gravity_axis_buttons.append(rb)
            if self.length_axis_var.get() == self.gravity_axis_var.get():
                if self.length_axis_var.get() != 'Z':
                    self.gravity_axis_var.set('Z')
                else:
                    self.gravity_axis_var.set('X')
        self.update_idletasks()

    # ------------------ 页面2 ------------------
    def setup_divide_page(self, parent):
        tk.Label(parent, text="网格分区参数：").grid(row=0, column=0, sticky='w', padx=5, pady=10)

        self.grid_frame = tk.Frame(parent)
        self.grid_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=10, sticky='w')

        tk.Label(self.grid_frame, text="管长方向等份数（u坐标）：").grid(row=0, column=0, padx=5)
        self.x_div_var = tk.StringVar(value='2')
        tk.Spinbox(self.grid_frame, from_=1, to=20, textvariable=self.x_div_var, width=5,
                   command=self.preview_division).grid(row=0, column=1, padx=5)

        tk.Label(self.grid_frame, text="重力方向等份数（v坐标）：").grid(row=0, column=2, padx=5)
        self.y_div_var = tk.StringVar(value='2')
        tk.Spinbox(self.grid_frame, from_=1, to=20, textvariable=self.y_div_var, width=5,
                   command=self.preview_division).grid(row=0, column=3, padx=5)

        remark_label1 = tk.Label(parent, text="1）请留意网格预览图，确保管长方向等份数调整边为横向边，若不是请点击“交换划分方向”；",
                                fg="gray", font=('微软雅黑', 8))
        remark_label1.grid(row=2, column=0, columnspan=4, sticky='w', padx=5, pady=1)
        
        remark_label2 = tk.Label(parent, text="2）平面内部划分上建立了u、v坐标，默认u为管长方向，v为重力方向。",
                                fg="gray", font=('微软雅黑', 8))
        remark_label2.grid(row=3, column=0, columnspan=4, sticky='w', padx=5, pady=1)

        self.swap_btn = tk.Button(parent, text="交换划分方向", command=self.toggle_swap_uv, bg="#e0e0e0")
        self.swap_btn.grid(row=4, column=1, pady=5)

        tk.Button(parent, text="预览划分线", command=self.preview_division, bg="#e0e0e0").grid(row=5, column=1, pady=5)

        self.division_viz_frame = tk.Frame(parent, height=300, width=650)
        self.division_viz_frame.grid(row=6, column=0, columnspan=4, padx=5, pady=10)
        self.division_viz_frame.grid_propagate(False)

        self.compute_btn = tk.Button(parent, text="执行分区与均值计算", command=self.compute_regions, bg="#e0e0e0")
        self.compute_btn.grid(row=7, column=1, pady=20)

        self.divide_status = tk.Label(parent, text="点击计算后才会呈现变动内容", fg="gray", font=('微软雅黑', 8))
        self.divide_status.grid(row=8, column=0, columnspan=4, pady=2)

    def toggle_swap_uv(self):
        self.swap_uv = not self.swap_uv
        self.swap_btn.config(text="恢复划分方向" if self.swap_uv else "交换划分方向")
        self.preview_division()
        # 不再自动调用 compute_regions()

    def preview_division(self):
        if self.processor is None or self.processor.geometry is None:
            return
        for widget in self.division_viz_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 5))

        if self.processor.quad_type == 'tilted':
            _, u_vals, v_vals = self.processor.geometry.get_plane_points()
            x = u_vals
            y = v_vals
            x_label = self.processor.length_axis + ' (投影)'
            y_label = self.processor.gravity_axis + ' (投影)'
            vertices = self.processor.geometry.vertices
        else:
            df_plane = self.processor.geometry.df_plane
            x_axis = self.processor.length_axis
            y_axis = self.processor.gravity_axis
            x = df_plane[x_axis].values
            y = df_plane[y_axis].values
            x_label = x_axis
            y_label = y_axis
            verts = self.processor.geometry.vertices
            source_axis1 = self.processor.geometry.x_col
            source_axis2 = self.processor.geometry.y_col
            verts_user = []
            for v in verts:
                val1 = v[0]
                val2 = v[1]
                x_val = val1 if source_axis1 == x_axis else val2 if source_axis2 == x_axis else None
                y_val = val2 if source_axis2 == y_axis else val1 if source_axis1 == y_axis else None
                if x_val is None or y_val is None:
                    x_val, y_val = v[0], v[1]
                verts_user.append(np.array([x_val, y_val]))
            vertices = verts_user

        ax.scatter(x, y, c='blue', s=5, alpha=0.3)
        poly_pts = np.vstack([vertices, vertices[0]])
        ax.plot(poly_pts[:,0], poly_pts[:,1], 'b-', linewidth=2, label='四边形边界')

        u_input = int(self.x_div_var.get())
        v_input = int(self.y_div_var.get())
        if not self.swap_uv:
            u_edges = np.linspace(0, 1, u_input + 1)
            v_edges = np.linspace(0, 1, v_input + 1)
        else:
            u_edges = np.linspace(0, 1, v_input + 1)
            v_edges = np.linspace(0, 1, u_input + 1)

        for u in u_edges:
            pts = []
            for v in np.linspace(0, 1, 50):
                p0, p1, p2, p3 = vertices
                xc = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
                yc = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
                pts.append([xc, yc])
            pts = np.array(pts)
            ax.plot(pts[:,0], pts[:,1], 'r--', linewidth=1.5, alpha=0.7)
        for v in v_edges:
            pts = []
            for u in np.linspace(0, 1, 50):
                p0, p1, p2, p3 = vertices
                xc = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
                yc = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
                pts.append([xc, yc])
            pts = np.array(pts)
            ax.plot(pts[:,0], pts[:,1], 'r--', linewidth=1.5, alpha=0.7)

        if not self.swap_uv:
            for i in range(u_input):
                uc = (u_edges[i] + u_edges[i+1]) / 2
                for j in range(v_input):
                    vc = (v_edges[j] + v_edges[j+1]) / 2
                    u_idx = i+1
                    v_idx = j+1
                    xc = (1-uc)*(1-vc)*vertices[0][0] + uc*(1-vc)*vertices[1][0] + uc*vc*vertices[2][0] + (1-uc)*vc*vertices[3][0]
                    yc = (1-uc)*(1-vc)*vertices[0][1] + uc*(1-vc)*vertices[1][1] + uc*vc*vertices[2][1] + (1-uc)*vc*vertices[3][1]
                    ax.text(xc, yc, f"{u_idx},{v_idx}", ha='center', va='center',
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
        else:
            for i in range(v_input):
                uc = (u_edges[i] + u_edges[i+1]) / 2
                for j in range(u_input):
                    vc = (v_edges[j] + v_edges[j+1]) / 2
                    u_idx = j+1
                    v_idx = i+1
                    xc = (1-uc)*(1-vc)*vertices[0][0] + uc*(1-vc)*vertices[1][0] + uc*vc*vertices[2][0] + (1-uc)*vc*vertices[3][0]
                    yc = (1-uc)*(1-vc)*vertices[0][1] + uc*(1-vc)*vertices[1][1] + uc*vc*vertices[2][1] + (1-uc)*vc*vertices[3][1]
                    ax.text(xc, yc, f"{u_idx},{v_idx}", ha='center', va='center',
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

        if self.invert_x:
            ax.invert_xaxis()
        if self.invert_y:
            ax.invert_yaxis()

        ax.set_title(f'网格预览 ({u_input}×{v_input})', fontsize=12)
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        canvas = FigureCanvasTkAgg(fig, self.division_viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.division_fig = fig

    # ------------------ 页面3 ------------------
    def setup_result_page(self, parent):
        self.result_text = tk.Text(parent, height=15, width=100, font=('Courier New', 9))
        self.result_text.pack(padx=5, pady=5, fill='both', expand=True)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="显示分区可视化图", command=self.show_division_viz, bg="#e0e0e0").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="显示结果矩阵", command=self.show_result_matrix, bg="#e0e0e0").pack(side=tk.LEFT, padx=5)

    # ------------------ 页面4 ------------------
    def setup_plot_page(self, parent):
        tk.Label(parent, text="点击下方按钮生成对比云图（原始数据三角剖分填充 vs 区域加权均值四边形填充）").pack(pady=10)
        tk.Button(parent, text="生成云图", command=self.generate_compare_plot, bg="#e0e0e0").pack(pady=10)

        self.plot_container = tk.Frame(parent)
        self.plot_container.pack(expand=True, fill='both', padx=20, pady=10)

        self.plot_frame = tk.Frame(self.plot_container, bg='white', relief='sunken', bd=1)
        self.plot_frame.place(relx=0.5, rely=0.5, anchor='center', width=800, height=450)

        self.plot_placeholder = tk.Label(self.plot_frame, text="点击「生成云图」按钮后，此处将显示对比云图",
                                         font=('微软雅黑', 10), fg='gray', bg='white')
        self.plot_placeholder.place(relx=0.5, rely=0.5, anchor='center')
        self.current_plot_canvas = None

    # ------------------ 交互逻辑 ------------------
    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if filepath:
            self.file_path_var.set(filepath)
            self.processor = FluentDataProcessor(filepath)
            if self.processor.df is not None:
                info = f"文件加载成功！\n数据行数：{len(self.processor.df)}\n列名：{list(self.processor.df.columns)}\n"
                info += f"识别坐标列：{self.processor.coord_cols}\n数值列：{self.processor.value_col}\n面积列：{self.processor.area_col}"
                self.geo_info_text.delete(1.0, tk.END)
                self.geo_info_text.insert(tk.END, info)
                self.update_direction_options()
            else:
                self.processor = None

    def identify_geometry(self):
        if self.processor is None:
            messagebox.showwarning("警告", "请先加载Excel文件")
            return

        quad_type = self.quad_type_var.get()
        if quad_type == 'plane':
            const_axis = self.const_axis_var.get()
            try:
                const_value = float(self.const_value_var.get())
            except:
                messagebox.showerror("错误", "等值坐标值必须是数字")
                return
            length_axis = self.length_axis_var.get()
            gravity_axis = self.gravity_axis_var.get()
            if length_axis == gravity_axis:
                messagebox.showerror("错误", "管长方向和重力方向不能相同")
                return
            try:
                params = self.processor.analyze_quadrilateral('plane', const_axis, const_value,
                                                              length_axis, gravity_axis)
                info = f"等值面四边形识别完成！\n等值面：{const_axis} = {const_value}\n"
                if params.get('fit_side_lengths'):
                    info += "拟合四边形边长: " + ", ".join([f"{s:.4f}" for s in params['fit_side_lengths']]) + "\n"
                info += f"拟合周长: {params['fit_perimeter']:.4f}\n"
                info += f"设置：管长方向 = {length_axis}，重力方向 = {gravity_axis}\n"
                self.geo_info_text.delete(1.0, tk.END)
                self.geo_info_text.insert(tk.END, info)
                self.visualize_shape()
                self.preview_division()
                messagebox.showinfo("完成", "四边形识别完成")
            except Exception as e:
                messagebox.showerror("错误", f"识别失败：{str(e)}")
        else:
            length_axis = self.length_axis_var.get()
            gravity_axis = self.gravity_axis_var.get()
            if length_axis == gravity_axis:
                messagebox.showerror("错误", "管长方向和重力方向不能相同")
                return
            try:
                params = self.processor.analyze_quadrilateral('tilted', None, None,
                                                              length_axis, gravity_axis)
                info = f"倾斜面四边形识别完成！\n拟合平面，基于投影点集。\n"
                info += f"投影基向量方向：管长方向沿 {length_axis} 轴投影，重力方向沿 {gravity_axis} 轴投影\n"
                if params.get('fit_side_lengths'):
                    info += "拟合四边形边长: " + ", ".join([f"{s:.4f}" for s in params['fit_side_lengths']]) + "\n"
                info += f"拟合周长: {params['fit_perimeter']:.4f}\n"
                self.geo_info_text.delete(1.0, tk.END)
                self.geo_info_text.insert(tk.END, info)
                self.visualize_shape()
                self.preview_division()
                messagebox.showinfo("完成", "四边形识别完成")
            except Exception as e:
                messagebox.showerror("错误", f"识别失败：{str(e)}")

    def visualize_shape(self):
        if self.processor is None or self.processor.geometry is None:
            return
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 5))

        if self.processor.quad_type == 'tilted':
            coord1 = self.processor.geometry.coord1_vals
            coord2 = self.processor.geometry.coord2_vals
            x_label = self.processor.length_axis + ' (投影)'
            y_label = self.processor.gravity_axis + ' (投影)'
            title = f'倾斜面四边形识别 (拟合平面)'
            x_data = coord1
            y_data = coord2
            verts = self.processor.geometry.vertices
        else:
            df_plane = self.processor.geometry.df_plane
            x_axis = self.processor.length_axis
            y_axis = self.processor.gravity_axis
            x_data = df_plane[x_axis].values
            y_data = df_plane[y_axis].values
            x_label = x_axis
            y_label = y_axis
            title = f'等值面四边形识别 (形心位于{self.processor.geometry.const_axis} = {self.processor.geometry.const_value})'
            verts = self.processor.geometry.vertices
            source_axis1 = self.processor.geometry.x_col
            source_axis2 = self.processor.geometry.y_col
            verts_user = []
            for v in verts:
                val1 = v[0]
                val2 = v[1]
                x_val = val1 if source_axis1 == x_axis else val2 if source_axis2 == x_axis else None
                y_val = val2 if source_axis2 == y_axis else val1 if source_axis1 == y_axis else None
                if x_val is None or y_val is None:
                    x_val, y_val = v[0], v[1]
                verts_user.append(np.array([x_val, y_val]))
            verts = verts_user

        ax.scatter(x_data, y_data, c='blue', s=10, alpha=0.5, label='数据点')
        if verts is not None:
            poly_pts = np.vstack([verts, verts[0]])
            ax.plot(poly_pts[:,0], poly_pts[:,1], 'b-', linewidth=2, label='拟合四边形')
            side_lengths = self.processor.geo_params.get('fit_side_lengths', [])
            if side_lengths and len(verts) == 4:
                for i in range(4):
                    p1 = verts[i]
                    p2 = verts[(i+1)%4]
                    mid = (p1 + p2) / 2
                    ax.text(mid[0], mid[1], f'{side_lengths[i]:.3f}',
                           ha='center', va='center', fontsize=8,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

        if self.invert_x:
            ax.invert_xaxis()
        if self.invert_y:
            ax.invert_yaxis()

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='best')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.fig = fig

    def compute_regions(self):
        if self.processor is None or self.processor.geometry is None:
            messagebox.showwarning("警告", "请先在页面1中识别四边形")
            return
        try:
            u_input = int(self.x_div_var.get())
            v_input = int(self.y_div_var.get())
            self.region_info = self.processor.divide_grid(u_input, v_input, self.swap_uv)
            self.result_df = self.processor.compute_mean(self.region_info)
            total_points = self.result_df['点数'].sum()
            total_row = pd.DataFrame([['总计', '', '', '', '', '', total_points]], 
                                     columns=self.result_df.columns)
            self.result_df = pd.concat([self.result_df, total_row], ignore_index=True)
            display_df = self.result_df[['区域名称', 'u范围', 'v范围', 'u索引', 'v索引', f'{self.processor.value_col}_加权均值', '点数']]
            self.result_text.delete(1.0, tk.END)
            if HAS_TABULATE:
                table = tabulate(display_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                table = display_df.to_string(index=False)
            self.result_text.insert(tk.END, table)
            messagebox.showinfo("完成", "分区与面积加权平均计算完成")
        except Exception as e:
            messagebox.showerror("错误", f"计算失败：{str(e)}")

    def show_division_viz(self):
        if self.processor is None or self.region_info is None:
            messagebox.showwarning("警告", "请先完成分区与均值计算")
            return
        DivisionVisualizationWindow(self, self.processor, self.region_info, 
                                    self.invert_x, self.invert_y)

    def show_result_matrix(self):
        if self.result_df is None:
            messagebox.showwarning("警告", "请先完成分区与均值计算")
            return
        df_no_total = self.result_df[self.result_df['区域名称'] != '总计'].copy()
        pivot = df_no_total.pivot_table(index='v索引', columns='u索引', values=f'{self.processor.value_col}_加权均值')
        # 根据四边形类型和方向调转状态排序
        if self.processor.quad_type == 'plane':
            # 等值面：原有规则
            col_ascending = self.invert_x   # 反向(True) -> 升序，正向(False) -> 降序
            row_ascending = not self.invert_y  # 正向(False) -> 升序，反向(True) -> 降序
        else:
            # 倾斜面：新规则
            # 管长方向（U索引，列）：正向(False) -> 正序（升序），反向(True) -> 倒序（降序）
            col_ascending = not self.invert_x
            # 重力方向（V索引，行）：正向(False) -> 倒序（降序），反向(True) -> 正序（升序）
            row_ascending = self.invert_y
        pivot = pivot.sort_index(axis=1, ascending=col_ascending)
        pivot = pivot.sort_index(axis=0, ascending=row_ascending)
        pivot.index.name = "V索引"
        pivot.columns.name = "U索引"
        MatrixResultWindow(self, pivot)

    def generate_compare_plot(self):
        if self.processor is None or self.result_df is None:
            messagebox.showwarning("警告", "请先完成分区与均值计算")
            return

        df_plane = self.processor.df_plane_labeled
        if df_plane is None:
            messagebox.showwarning("警告", "没有可用的平面数据")
            return

        if self.processor.quad_type == 'tilted':
            x = df_plane['U_proj'].values
            y = df_plane['V_proj'].values
            x_label = self.processor.length_axis + ' (投影)'
            y_label = self.processor.gravity_axis + ' (投影)'
            title_note = "(拟合平面投影)"
            vertices = self.processor.geometry.vertices
        else:
            x_axis = self.processor.length_axis
            y_axis = self.processor.gravity_axis
            x = df_plane[x_axis].values
            y = df_plane[y_axis].values
            x_label = x_axis
            y_label = y_axis
            title_note = f"({self.processor.geometry.const_axis} = {self.processor.geometry.const_value})"
            verts = self.processor.geometry.vertices
            source_axis1 = self.processor.geometry.x_col
            source_axis2 = self.processor.geometry.y_col
            verts_user = []
            for v in verts:
                x_val = v[0] if source_axis1 == x_axis else v[1] if source_axis2 == x_axis else None
                y_val = v[1] if source_axis2 == y_axis else v[0] if source_axis1 == y_axis else None
                if x_val is None or y_val is None:
                    x_val, y_val = v[0], v[1]
                verts_user.append(np.array([x_val, y_val]))
            vertices = verts_user

        z = df_plane[self.processor.value_col].values
        tri_obj = tri.Triangulation(x, y)

        u_edges = self.processor.u_edges
        v_edges = self.processor.v_edges
        swap = self.processor.swap_uv

        df_no_total = self.result_df[self.result_df['区域名称'] != '总计'].copy()
        region_means = {}
        for idx, row in df_no_total.iterrows():
            u_idx = row['u索引']
            v_idx = row['v索引']
            region_means[(u_idx, v_idx)] = row[f'{self.processor.value_col}_加权均值']

        def get_quad_vertices(u_idx, v_idx):
            if not swap:
                i = u_idx - 1
                j = v_idx - 1
                u_low = u_edges[i]
                u_high = u_edges[i+1]
                v_low = v_edges[j]
                v_high = v_edges[j+1]
            else:
                i = v_idx - 1
                j = u_idx - 1
                u_low = u_edges[i]
                u_high = u_edges[i+1]
                v_low = v_edges[j]
                v_high = v_edges[j+1]
            p0, p1, p2, p3 = vertices
            p00 = (1-u_low)*(1-v_low)*p0 + u_low*(1-v_low)*p1 + u_low*v_low*p2 + (1-u_low)*v_low*p3
            p01 = (1-u_low)*(1-v_high)*p0 + u_low*(1-v_high)*p1 + u_low*v_high*p2 + (1-u_low)*v_high*p3
            p11 = (1-u_high)*(1-v_high)*p0 + u_high*(1-v_high)*p1 + u_high*v_high*p2 + (1-u_high)*v_high*p3
            p10 = (1-u_high)*(1-v_low)*p0 + u_high*(1-v_low)*p1 + u_high*v_low*p2 + (1-u_high)*v_low*p3
            return np.array([p00, p01, p11, p10])

        all_means = list(region_means.values())
        vmin = min(all_means) if all_means else 0
        vmax = max(all_means) if all_means else 1
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap('jet')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

        tcf = ax1.tripcolor(tri_obj, z, shading='gouraud', cmap='jet')
        ax1.set_title(f'原始数据分布 {title_note}', fontsize=11, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=9)
        ax1.set_ylabel(y_label, fontsize=9)
        cbar1 = plt.colorbar(tcf, ax=ax1)
        cbar1.set_label(self.processor.value_col, fontsize=9)

        for (u_idx, v_idx), mean_val in region_means.items():
            color = cmap(norm(mean_val))
            quad_verts = get_quad_vertices(u_idx, v_idx)
            patch = Polygon(quad_verts, closed=True, facecolor=color, edgecolor='black', linewidth=0.3, alpha=0.9)
            ax2.add_patch(patch)

        if self.invert_x:
            ax1.invert_xaxis()
            ax2.invert_xaxis()
        if self.invert_y:
            ax1.invert_yaxis()
            ax2.invert_yaxis()

        ax2.set_title(f'区域加权均值填充图 {title_note}', fontsize=11, fontweight='bold')
        ax2.set_xlabel(x_label, fontsize=9)
        ax2.set_ylabel(y_label, fontsize=9)
        ax2.autoscale()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax2)
        cbar2.set_label(f'{self.processor.value_col} 加权均值', fontsize=9)

        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.current_plot_canvas = canvas


if __name__ == "__main__":
    app = Application()
    app.mainloop()