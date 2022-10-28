import math
import numpy as np

def configure_TES(
        d_TES=10,
        h_TES=35.72,
        s_TES_wall=0.005,
        s_TES_iso=0.2,
        lambda_TES_wall=50,
        lambda_TES_iso=0.032,
        alpha_TES_in=1500,
        alpha_TES_out=15,
        T_amb = np.array([10]),
        T_TES_max=90,
        T_TES_min=45,
        cp_water=4195,
        rho_water=1000,
):
        """

        Parameters
        ----------
        d_TES : float
                Diameter of TES [m]
        h_TES : float
                Height of TES [m]
        s_TES_wall : float
                Thickness of TES wall [m]
        s_TES_iso : float
                Thickness of isolation layer [m]
        lambda_TES_wall : float
                Heat conductivity of wall material [W/(m*K)]
        lambda_TES_iso : float
                Heat conductivity of isolation material [W/(m*K)]
        alpha_TES_in : float
                Heat transfer coefficient inside [W/(m2*K)]
        alpha_TES_out : float
                Heat transfer coefficient outside [W/(m2*K)]
        T_amb
                Environment temperature timeseries [°C]
        T_TES_max : float
                Maximal storage temperature [°C]
        T_TES_min : float
                Minimal storage temperature [°C]
        cp_water : float
                Heat capacity of storage medium [J/(kg*K)]
        rho_water : float
                Density of storage medium [kg/m3]

        Returns
        -------
        Q_TES_load: float
                Maximal storage capacity [kWh]
        gamma_qdot_loss
                Fixed losses as share of nominal storage capacity [-]
        beta_qdot_loss: float
                Relative loss of storage content within one timestep [-]
        """

        "Storage volume [m3]"
        V_TES = (d_TES * 0.5) ** 2 * math.pi * h_TES

        "top and bottom surfaces [m2]"
        A_TES_floor = (d_TES * 0.5) ** 2 * math.pi

        "lateral surface surfaces [m2]"
        A_TES_lateral_in = d_TES * h_TES * math.pi

        "Storage surface [m2]"
        A_TES = A_TES_lateral_in+2*A_TES_floor

        "Thermal transmittance [W/K]"
        UxA = A_TES/(1/alpha_TES_in+s_TES_wall/lambda_TES_wall+s_TES_iso/lambda_TES_iso+1/alpha_TES_out)


        Q_TES_load = ((T_TES_max-T_TES_min)*V_TES*cp_water*rho_water)/(3600*1000)
        beta_qdot_loss = UxA * (T_TES_max - T_TES_min)/Q_TES_load

        if len(T_amb) == 1:
            gamma_qdot_loss = UxA*(T_TES_min-T_amb[0])/Q_TES_load
        else:
            gamma_qdot_loss = UxA*(T_TES_min-T_amb)/Q_TES_load

        return Q_TES_load, gamma_qdot_loss, beta_qdot_loss

if __name__ == '__main__':
    a,b,c = configure_TES(T_amb=np.array([10]))




