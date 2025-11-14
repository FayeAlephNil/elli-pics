from manim import *
from polys import stereographic_proj
import polys
from sympy import symbols, Poly
import numpy as np
import csv
import json
from manim.utils import space_ops

class StereographicTest(ThreeDScene):
    def construct(self):
        x = 0.1
        y = 0.3

        axes = ThreeDAxes()
        sphere = Sphere([0,0,0],color=BLUE)
        point = Dot3D([x, y, 0],color=GREEN)
        top = Dot3D([0,0,1],color=GREEN)
        
        line_start = Dot3D([-1*x,-1*y,2])
        line_end = Dot3D([2*x,2*y,-1])

        line = Line3D(line_start, line_end,color=RED)
        projed_point = Dot3D(stereographic_proj(complex(x,y)))

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(axes))
        self.play(Create(sphere))
        print("Meow")
        self.wait()
        self.add(point, top)
        self.wait()
        self.play(Create(line))
        self.wait()
        self.add(projed_point)
        self.wait()

class TestPlotDivisor(ThreeDScene):
    def construct(self):
        s,t = symbols('s t')
        divisor = Poly(s**(12) - t**(12),s,t)
        rad = 2

        roots, top = polys.find_zeros_on_P1(divisor)
        sphere = Sphere([0,0,0],radius=rad,color=BLUE)
        points = create_points_from_divisor(divisor)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(sphere))

        self.play(*[Create(pt) for pt in points])
        self.wait()
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(5)
        self.stop_ambient_camera_rotation()

class AnimatePointsOnSphere(ThreeDScene):
    def load_points(file_name):
        data = []
        with open(file_name,'r') as myfile:
            reader = csv.reader(filter(lambda row: row[0] != '#', myfile))
            for row in reader:
                data.append([json.loads(l) for l in row])
        return data

    def check_front(pt,phi,theta,rad=1, eps=1e-10):
        cam = space_ops.spherical_to_cartesian([rad,theta,phi])
        return np.dot(pt,cam) > eps


    def data_to_dots(data_divisor,r=1,phi=0,theta=0,color=RED):
        lst = []
        for pt in data_divisor:
            dot = Dot3D([r*pt[0],r*pt[1],r*pt[2]])

            if AnimatePointsOnSphere.check_front(dot.get_center(), phi,theta,rad=r):
                dot.set_opacity(1.0)
            else:
                dot.set_opacity(0.2)

            lst.append(dot)
        return lst

    def single_transform(old,new):
        lst = []
        new_cpy = new.copy()
        for dot1 in old:
            winner = None
            winner_dist = -1
            for dot2 in new_cpy:
                displace = dot1.get_center()-dot2.get_center()
                new_dist = np.dot(displace, displace)
                if (new_dist < winner_dist) or winner_dist == -1:
                    winner = dot2
                    winner_dist = new_dist
            new_cpy.remove(winner)
            lst.append(ReplacementTransform(dot1,winner,rate_func=linear))
        return lst

    def create_pts(dots):
        return [Create(d) for d in dots]

    def construct(self):
        rad = 2
        data = AnimatePointsOnSphere.load_points('data/hesse-degeneration')
        num  = 100
        step = 2
        total_time = 10
        assert step*num <= 200, "Not enough data"
        

        sphere = Sphere([0,0,0],radius=rad)
        sphere.set_opacity(0.5)
        sphere.set_color(BLUE)
        phi = 75*DEGREES
        theta = 30*DEGREES

        self.set_camera_orientation(phi=phi, theta=theta)
        self.play(Create(sphere))
        self.wait()

        all_dots = [AnimatePointsOnSphere.data_to_dots(d,r=rad,phi=phi,theta=theta) for d in data[0:num*step:step]]
        initial = AnimatePointsOnSphere.create_pts(all_dots[0])
        self.play(*initial)
        self.wait()

        # Attempt a single transform
        for i in range(0,num-1):
            trans = AnimatePointsOnSphere.single_transform(all_dots[i],all_dots[i+1])
            print(trans)
            self.play(trans,run_time = total_time/(num-1))



def create_points_from_divisor(div,radius=1,color=RED, rescale_cx=True):
    roots, top = polys.find_zeros_on_P1(div)
    if rescale_cx:
        val = radius
    else:
        val = 1
    points = [Dot3D(stereographic_proj(val*root,r=radius),color=color) for root in roots]
    if top:
        points.append(Dot3D([0,0,radius], color=color))

    return points

class TestCubicPencil(ThreeDScene):
    def construct(self):
        x,y = symbols('x y')
        cubic1 = Poly(y**2 - x**3 - x + 0.3,x,y)
        cubic2 = Poly(y**2 - x**3 + complex(0,0.2)*x - complex(0.1,0.5),x,y)
        divisor1, divisor2 = polys.j_cubic_pencil(cubic1, cubic2)

        rad = 2 
        points_0 = create_points_from_divisor(divisor1, radius=rad, color=GREEN)
        points_infty = create_points_from_divisor(divisor2, radius=rad, color=RED)

        sphere = Sphere([0,0,0], radius=2,color=BLUE)

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.play(Create(sphere))

        self.play(*[Create(pt) for pt in points_0 + points_infty])
        self.wait()
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(10)
        self.stop_ambient_camera_rotation()

class CubicPencilReal(Scene):
    def construct(self):
        x_range = (-6,6,1)
        y_range = (-6,6,1)
        func1 = lambda x,y: -x+(y-2)**3+4*(y-2)**2-5
        func2 = lambda x,y: 5+y-x**3-4*x**2
        graph1 = ImplicitFunction(
            func1,
            x_range=x_range,
            y_range=y_range,
            color=BLUE
        )
        graph1.scale(0.9)
        graph2 = ImplicitFunction(
            func2,
            x_range=x_range,
            y_range=y_range,
            color=RED
        )
        graph2.scale(0.9)
        tracker = ValueTracker(0)
        graph_changing = ImplicitFunction(
            lambda x, y: tracker.get_value()*func1(x,y) + func2(x,y),
            x_range=x_range,
            y_range=y_range,
            color=GREEN
        ).scale(0.9)
        graph_changing.add_updater(lambda t: t.become(
            ImplicitFunction(
                lambda x, y: tracker.get_value()*func1(x,y) + func2(x,y),
                x_range = x_range,
                y_range = y_range,
                color = GREEN
            ).scale(0.9)
        ))

        plane = NumberPlane(x_range = x_range, y_range = y_range)
        label = MathTex(f"{tracker.get_value():.2f}" + r"P+Q").to_corner(DR) 
        label.add_updater(lambda t: t.become(
            MathTex(f"{tracker.get_value():.2f}" + r"P+Q", color=GREEN).to_corner(DR)
        ))

        self.add(plane,graph1,graph2)
        self.wait()
        self.add(tracker,graph_changing,label)
        self.play(tracker.animate(rate_func=rate_functions.ease_in_sine).set_value(5),run_time=3)
        self.play(tracker.animate(rate_func=rate_functions.ease_out_sine).set_value(0),run_time=3)
        self.play(tracker.animate(rate_func=rate_functions.ease_in_sine).set_value(-5),run_time=3)


def main():
    with tempconfig({"quality": "medium_quality", "preview": True}):
        scene = StereographicTest()
        scene.render()
    print("Hello from manimations!")

if __name__ == "__main__":
    main()
