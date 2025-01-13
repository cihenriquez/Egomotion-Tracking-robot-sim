import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Configurar Matplotlib para usar LaTeX en notación matemática
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    # Cargar el archivo .npy
    file_name = "simdata.npy"
    if not os.path.exists(file_name):
        print(f"Error: The file '{file_name}' does not exist in the current directory.")
        return

    try:
        data = np.load(file_name)
    except Exception as e:
        print(f"Error loading the file: {e}")
        return

    # Verificar que el archivo tenga un formato esperado (2D array)
    if data.ndim != 2 or data.shape[1] < 19:
        print("Error: The file must contain a 2D array with at least 19 columns.")
        return

    # Descriptive names for each column
    
    column_names = [
        "Time [s]",
        "Position in x [m]",
        "Position in y [m]",
        "Angle [rad]",
        "True value",
        "True value",
        "Reference position in x [m]",
        "Reference position in y [m]",
        "Reference angle [rad]",
        "Right wheel",
        "Left wheel",
        "True value",
        "From simulated sensor",
        "True value",
        "From simulated sensor",
        "From camera data estimation",
        "From camera data estimation",
        "From camera data estimation",
        "From camera data estimation",
        "Estimated from simulated IMU",
        "Estimated from simulated IMU",
        "Estimated from simulated IMU",
        "Estimated from simulated IMU",
        "Est. using cameras data velocities and IMU accel",
        "Est. using cameras data velocities and IMU accel",
        "Est. using cameras data velocities and IMU accel",
        "Est. using cameras data velocities and IMU accel",
        "Estimated using only cameras data",
        "Estimated using only cameras data",
        "Estimated using only cameras data",
        "Estimated using only cameras data",
        "True value",
        "True value",
        "True value",
        "True value"
    ]

    # Categories of graphs
    categories = {
        #"Position": [1, 2, 6, 7, 8],
        "Linear Velocity": [4, 15],
        "Angular Velocity": [5, 16],
        "Linear Acceleration": [11, 17],
        "Angular Acceleration": [13, 18],
        "Torque": [9, 10]
    }

    # Y-axis labels by category
    y_labels = {
        "Position": "Values [m | rad]",
        "Linear Velocity": "Linear velocity " + r'$[\frac{m}{s} ]$',
        "Angular Velocity": "Angular velocity " + r'$[\frac{rad}{s} ]$',
        "Linear Acceleration": "Linear acceleration "+ r"$[ \frac{m}{s^2} ]$",
        "Angular Acceleration": "Angular acceleration "+ r"$[ \frac{rads}{s^2} ]$",
        "Torque": "Torque [N$\\cdot$m]"
    }

    # Use the first column as time and filter out rows with time == 0
    time = data[:, 0]
    valid_indices = time > 0
    if not np.any(valid_indices):
        print("Error: No data with time greater than 0.")
        return

    data = data[valid_indices]
    time = data[:, 0]

    def save_with_zoom(fig, ax, filename):
        """Guarda la figura manteniendo el zoom aplicado manualmente."""
        # Obtener los límites actuales de los ejes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Aplicar los límites antes de guardar
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Guardar la figura
        fig.savefig(filename)
        print(f"Saved zoomed plot as {filename}")

    # Plot each category
    i=4
    for category, columns in categories.items():
        fig, ax = plt.subplots()
        for col_idx in columns:
            ax.plot(time, data[:, col_idx], label=column_names[col_idx], linewidth=1, zorder=i)
            i-=1
        ax.set_xlabel(column_names[0])
        ax.set_ylabel(y_labels[category])
        #ax.set_title(f"{category} vs Time")
        ax.legend()
        ax.grid(True)

        # Mostrar y esperar que el usuario ajuste el zoom
        plt.show()

        # Guardar con el zoom ajustado
        save_with_zoom(fig, ax, f"plot_{category.replace(' ', '_')}.pdf")

    # Indices for the variables
    variables = {
        "Mass [kg]": (19, 23,27,31),
        "Linear Viscous Friction Coefficient "+ r"$[ \frac{N}{m/s} ]$": (20, 24,28,32),
        "Inertia Moment "+ r"$[ kg \cdot m^2 ]$": (21,25, 29,33),
        "Rotational Viscous Friction Coefficient "+ r"$[ \frac{N \cdot m}{rads/s} ]$": (22,26,30 ,34)
    }

    # Plot each variable
    for var_name, (est_idx,est1_idx,est2_idx, actual_idx) in variables.items():
        fig, ax = plt.subplots()
        ax.plot(time, data[:, est_idx],label=column_names[est_idx], linewidth=1,zorder=1)
        ax.plot(time, data[:, est1_idx],label=column_names[est1_idx], linewidth=1, zorder=2)
        ax.plot(time, data[:, est2_idx],label=column_names[est2_idx], linewidth=1,zorder=0)
        ax.plot(time, data[:, actual_idx], label=column_names[actual_idx],linewidth=2,linestyle='--',zorder=3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{var_name}")
        #ax.set_title(f"{var_name} Over Time")
        ax.legend()
        ax.grid(True)

        # Mostrar y esperar que el usuario ajuste el zoom
        plt.show()

        # Guardar con el zoom ajustado
        safe_name = var_name.replace(" ", "_").replace("/", "_")
        save_with_zoom(fig, ax, f"plot_{safe_name}.pdf")

if __name__ == "__main__":
    main()