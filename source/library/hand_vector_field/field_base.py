import numpy as np


class FieldBase:

    line_width = 0.05
    top_magnitude = 1.0

    def __init__(self, p1: tuple, p2: tuple, img_dims: tuple):
        if len(p1) != 2 or len(p2) != 2:
            raise ValueError
        img_dims = img_dims[:2]
        self.field = np.zeros(shape=img_dims+(2,), dtype=np.float32)

        self.p1 = np.array(p1)
        self.p2 = np.array(p2)

        self.produce_field()

    def produce_field(self):
        diff = self.p2 - self.p1
        if diff[0]:
            counter_delta = FieldBase.line_width * np.linalg.norm(diff) / abs(diff[0])
        else:
            counter_delta = self.field.shape[1]

        base_x_norm = max(min(self.p1[0], self.p2[0]) - FieldBase.line_width, .0)
        end_x_norm = min(max(self.p1[0], self.p2[0]) + FieldBase.line_width, 1.)

        steps_x = int((end_x_norm-base_x_norm)*self.field.shape[0])
        steps_y = int((2 * counter_delta) * self.field.shape[1])
        base_counter_start = int(base_x_norm * self.field.shape[0])

        if diff[0]:
            m = diff[1]/diff[0]
        else:
            m = 2 * self.field.shape[0] + 1

        if diff[1]:
            m_perp = - diff[0]/diff[1]
        else:
            m_perp = - 2 * self.field.shape[1] - 1

        standard_direction = np.array((1., m))
        standard_direction = standard_direction / np.linalg.norm(standard_direction)
        q = self.p1[1] - m * self.p1[0]

        line_normalization_factor = np.sqrt(1 + m**2 + q**2)

        tmp_p = np.array((0., 0.))

        for i in range(base_counter_start, min(base_counter_start+steps_x, self.field.shape[0]-1)):
            tmp_p[0] = i / self.field.shape[0]
            central_y = m * tmp_p[0] + q

            base_y = int(max(central_y - counter_delta, 0) * self.field.shape[1])
            for j in range(base_y, min(self.field.shape[1]-1, base_y + steps_y)):
                tmp_p[1] = j / self.field.shape[1]
                # perpendicular line passing through (x, y): y = m_perp * x + q_perp
                q_perp = tmp_p[1] - m_perp * tmp_p[0]
                # if p1 and p2 are on the same side of the perp line, then use the smaller distance
                if (m_perp * self.p1[0] + q_perp - self.p1[1])*(m_perp * self.p2[0] + q_perp - self.p2[1]) > 0:
                    dist = min(np.linalg.norm(tmp_p-self.p1), np.linalg.norm(tmp_p-self.p2))
                # else use the line distance
                else:
                    dist = abs((central_y - tmp_p[1])/line_normalization_factor)
                magnitude = max(0, FieldBase.top_magnitude * (1 - dist / FieldBase.line_width))
                self.field[i, j] = standard_direction * magnitude

