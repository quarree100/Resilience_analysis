import pandas as pd
import numpy as np
import os
import math


def configure_TES(
        d_TES=10,
        h_TES=35.72,
        s_TES_wall=0.005,
        s_TES_iso=0.2,
        lambda_TES_wall=50,
        lambda_TES_iso=0.032,
        alpha_TES_in=1500,
        alpha_TES_out=15,
        T_amb=10,
        T_TES_max=90,
        T_TES_min=45,
        cp_water=4195,
        rho_water=1000,
):
        """

        Parameters
        ----------
        d_TES : float
                diameter of TES in meter
        h_TES
        s_TES_wall
        s_TES_iso
        lambda_TES_wall
        lambda_TES_iso
        alpha_TES_in
        alpha_TES_out
        T_amb
        T_TES_max
        T_TES_min
        cp_water
        rho_water

        Returns
        -------

        """

        V_TES = (d_TES * 0.5) ** 2 * math.pi * h_TES
        A_TES_floor = (d_TES * 0.5) ** 2 * math.pi
        A_TES_lateral_in = d_TES * h_TES * math.pi
        A_TES = A_TES_lateral_in+2*A_TES_floor
        UxA = A_TES/(1/alpha_TES_in+s_TES_wall/lambda_TES_wall+s_TES_iso/lambda_TES_iso+1/alpha_TES_out)
        Q_TES_load = ((T_TES_max-T_TES_min)*V_TES*cp_water*rho_water)/(3600*1000)
        Qdot_loss_fix = UxA*(T_TES_min-T_amb)
        qdot_loss_rate = UxA * (T_TES_max - T_TES_min)/Q_TES_load

        return Qdot_loss_fix, qdot_loss_rate

if __name__ == '__main__':
    a,b = configure_TES()




