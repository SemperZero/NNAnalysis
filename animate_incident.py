# python -m pip install --upgrade pip
# pip install manim
# https://miktex.org/download
# manim -pql --frame_rate 60 --resolution 1778,1000 animate_incident.py TimestampScene




from manim import *
import numpy as np

class TimestampScene(Scene):
    def generate_class_points(self, center, count, std_dev=7.0):
        return np.random.normal(loc=center, scale=std_dev, size=count)

    def construct(self):
        np.random.seed(42)

        axis = NumberLine(
            x_range=[0, 60, 5],
            length=12,
            include_numbers=True,
            include_tip=True,
            decimal_number_config={"num_decimal_places": 1},
        )
        axis.move_to(ORIGIN)
        self.play(Create(axis))

        # Legend data: colors mapped to example VM connection errors
        legend_items = [
            (RED, "Timeout errors"),
            (BLUE, "Authentication failures"),
            (GREEN, "Network unreachable"),
            (ORANGE, "Disk I/O errors"),
        ]

        legend_group = VGroup()

        for i, (color, label) in enumerate(legend_items):
            dot = Dot(color=color, radius=0.12)
            dot.move_to(ORIGIN + UP * (3 - i * 0.6) + RIGHT * 3.3)

            text = Text(label, font="Consolas", font_size=20)
            # Position text horizontally right and vertically centered with dot
            text.next_to(dot, RIGHT, buff=0.15)
            text.align_to(dot, UP)  # align vertical center (dot and text baseline)

            legend_group.add(dot, text)

        self.play(*[Write(mob) for mob in legend_group])

        class_configs = [
            {"center": 5, "count": 4, "color": RED},
            {"center": 15, "count": 5, "color": BLUE},
            {"center": 30, "count": 16, "color": GREEN},
            {"center": 50, "count": 7, "color": ORANGE},
        ]

        all_points = []
        for config in class_configs:
            timestamps = self.generate_class_points(config["center"], config["count"])
            for t in timestamps:
                all_points.append({
                    "timestamp": float(t),
                    "color": config["color"]
                })

        all_points.sort(key=lambda x: x["timestamp"])

        annotations = {
            20: "sev3 aegis",
            30: "client complained",
            40: "sev2 aegis",
        }
        arrow_start_offsets = {
            20: UP * 2,
            30: UP * 1,
            40: UP * 2,
        }
        arrow_colors = {
            20: ORANGE,
            30: RED,
            40: RED,
        }

        added_annotations = set()

        for point_data in all_points:
            t = point_data["timestamp"]
            color = point_data["color"]

            for ann_x in sorted(annotations):
                if ann_x <= t and ann_x not in added_annotations:
                    self.wait(0.5)

                    arrow_tip = axis.number_to_point(ann_x)
                    start_offset = arrow_start_offsets.get(ann_x, UP * 1.5)
                    arrow_start = arrow_tip + start_offset

                    arrow = Arrow(
                        start=arrow_start,
                        end=arrow_tip,
                        buff=0,
                        stroke_width=3,
                        color=arrow_colors.get(ann_x)
                    )

                    label = Text(annotations[ann_x], font="Consolas", font_size=24)
                    label.next_to(arrow_start, UP if start_offset[1] > 0 else DOWN)

                    self.play(GrowArrow(arrow), FadeIn(label))
                    self.wait(0.5)

                    added_annotations.add(ann_x)

            dot = Dot(point=axis.number_to_point(t), color=color, radius=0.13, fill_opacity=0.75)
            self.play(GrowFromCenter(dot), run_time=0.5)
            self.wait(0.05)

        self.wait(2)
