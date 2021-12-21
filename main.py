# -*- coding: utf-8 -*-
"""
Calculates the decay constants and half-lives of 79-Sr and 79-Rb by minimizing
the chi squared of a two parameter fit (the decay constants of 79-Sr and 79-Rb)
for the activity (TBq) of 79-Rb against time in seconds.

Can:
 - use N-number of raw data files (in the format: | Time (hours) | Activity (TBq) | Activity Error (TBq) |)
   by adding the relative filepath to the 'data_filepaths' array.
 - display N-different relative chi squared differences by modifying the
   'key_chi_squared_differences' array.
 - change the sample size by varying the number of molecules held in the constant
   'number_of_molecules_in_sample'
 - change the relative save filepath and filename for the main plot by changing
   the constant 'main_plot_save_filepath'.
 - change the relative save filepath and filename for the contour plot by
   changing the constant 'contour_plot_save_filepath'.

Ben Gavan - 09/12/19
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import fmin
import sys
import os.path


# MARK: - Constants
number_of_molecules_in_sample = 6.022 * pow(10, 17)

key_chi_squared_differences = [
    1,
    2.30,
    5.99,
    9.21
]

data_filepaths = [
    '../Nuclear_data_1.csv',
    '../Nuclear_data_2.csv',
]

main_plot_save_filepath = 'BenGavan_main-plot.png'
contour_plot_save_filepath = 'BenGavan_contour-plot.png'

sr_decay_constant_start_value = 0.05
rb_decay_constant_start_value = 0.005


# MARK: - Entry point of the script.
def main():
    """
    Start point of the script.
    Returns
    ------
    None
    """
    times, activities, activity_errors = get_data()

    times = times * 60 * 60  # Hours -> Seconds

    sr_decay_constant = sr_decay_constant_start_value
    rb_decay_constant = rb_decay_constant_start_value

    # Calculate min fit including outliers
    sr_decay_constant, rb_decay_constant = calculate_minimum_fit(times, activities, activity_errors, sr_decay_constant,
                                                                 rb_decay_constant)

    # Second min fit with errors removed
    times, activities, activity_errors = remove_outliers(times, activities, activity_errors, sr_decay_constant,
                                                         rb_decay_constant)

    sr_decay_constant, rb_decay_constant = calculate_minimum_fit(times, activities, activity_errors, sr_decay_constant,
                                                                 rb_decay_constant)

    contours = plot_chi_squared_contour(times, activities, activity_errors, sr_decay_constant, rb_decay_constant)
    sr_decay_constant_uncertainty, rb_decay_constant_uncertainty = calculate_uncertainty_from_contour_plot(contours)

    plot_data(times, activities, activity_errors, sr_decay_constant, sr_decay_constant_uncertainty, rb_decay_constant,
              rb_decay_constant_uncertainty)


# MARK: - Reading and Validating data.
def get_data():
    """
    Gets the raw data for all of the given relative file paths.
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    """
    print_separator('Getting data.')

    check_files_exist()

    times = np.array([])
    activities = np.array([])
    activity_errors = np.array([])

    for filepath in data_filepaths:
        time, activity, activity_error = get_data_from_filepath(filepath)
        times = np.append(times, time)
        activities = np.append(activities, activity)
        activity_errors = np.append(activity_errors, activity_error)

    times, activities, activity_errors = validate_data(times, activities, activity_errors)

    return times, activities, activity_errors


def get_data_from_filepath(filepath):
    """
    Gets the raw data for the given filepath also removes the comments.
    Parameters
    --------
    filepath : str
    Returns
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    """
    print('Getting data for: {}'.format(filepath))

    data = np.genfromtxt(filepath, comments='%', delimiter=',', dtype=float)

    times = data[:, 0]
    activities = data[:, 1]
    activity_errors = data[:, 2]

    return times, activities, activity_errors


def check_files_exist():
    """
    Checks that all specified files exist (Ends program is any file does not exist).
    Returns
    ------
    None
    """
    is_file_non_exist = False
    for filepath in data_filepaths:
        if not os.path.isfile(filepath):
            print('ERROR: {} does not exist.'.format(filepath))
            is_file_non_exist = True

    if is_file_non_exist:
        print('Program terminating due to missing files.')
        sys.exit()


def validate_data(times, activities, activity_errors):
    """
    Validates all of the data subject to:
     - times greater than 0 and not 'nan'
     - activities greater than  and not 'nan'
     - activity_errors greater than and not 'nan'
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    Returns
    --------
    new_times : numpy.ndarray
    new_activities : numpy.ndarray
    new_activity_errors : numpy.ndarray
    """
    print_separator('Validating data')
    new_times = np.array([])
    new_activities = np.array([])
    new_activity_errors = np.array([])

    for time, activity, activity_error in zip(times, activities, activity_errors):
        if not is_greater_than_zero(time):
            continue
        if not is_greater_than_zero(activity):
            continue
        if not is_greater_than_zero(activity_error):
            continue

        new_times = np.append(new_times, time)
        new_activities = np.append(new_activities, activity)
        new_activity_errors = np.append(new_activity_errors, activity_error)

    return new_times, new_activities, new_activity_errors


def is_greater_than_zero(value):
    """
    Checks that the value is not a 'nan' and is physically acceptable (value is greater than zero).
    Parameters
    --------
    value : float
    Returns
    --------
    is_valid : bool
    """
    if np.isnan(value):
        return False
    if value <= 0:
        return False
    return True


def remove_outliers(times, activities, activity_errors, sr_decay_constant, rb_decay_constant):
    """
    Removes outlying data points if they are over 3 * error away from the minimized fit.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    new_times : numpy.ndarray
    new_activities : numpy.ndarray
    new_activity_errors : numpy.ndarray
    """
    print_separator('Removing outliers')

    new_times = np.array([])
    new_activities = np.array([])
    new_activity_errors = np.array([])

    for time, activity, activity_error in zip(times, activities, activity_errors):
        difference = np.abs(calculate_activity_rb(time, sr_decay_constant, rb_decay_constant) - activity)
        if difference <= activity_error * 3:
            new_times = np.append(new_times, time)
            new_activities = np.append(new_activities, activity)
            new_activity_errors = np.append(new_activity_errors, activity_error)

    return new_times, new_activities, new_activity_errors


# MARK: - Plotting
def plot_data(times, activities, activities_error, sr_decay_constant, sr_decay_constant_uncertainty, rb_decay_constant,
              rb_decay_constant_uncertainty):
    """
    Plots the raw data and minimized fit.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activities_error : numpy.ndarray
    sr_decay_constant : float
    sr_decay_constant_uncertainty : float
    rb_decay_constant : float
    rb_decay_constant_uncertainty : float
    Returns
    --------
    None
    """
    print_separator('Plotting activity-time plot')

    reduced_chi2 = calculate_reduced_chi_2(times, activities, activities_error, sr_decay_constant, rb_decay_constant)

    xs = np.linspace(min(times), max(times), 10000)

    residuals = calculate_residuals(times, activities, sr_decay_constant, rb_decay_constant)

    # Format decay constant values
    sr_decay_constant_string, sr_decay_constant_uncertainty_string = format_value_with_uncertainty(sr_decay_constant,
                                                                                                   sr_decay_constant_uncertainty, 3)
    rb_decay_constant_string, rb_decay_constant_uncertainty_string = format_value_with_uncertainty(rb_decay_constant,
                                                                                                   rb_decay_constant_uncertainty, 3)

    # Half-life calculations.
    sr_half_life_seconds, sr_half_life_uncertainty_seconds = calculated_half_life(sr_decay_constant,
                                                                                  sr_decay_constant_uncertainty)

    rb_half_life_seconds, rb_half_life_uncertainty_seconds = calculated_half_life(rb_decay_constant,
                                                                                  rb_decay_constant_uncertainty)

    sr_half_life_minutes = sr_half_life_seconds / 60
    sr_half_life_uncertainty_minutes = sr_half_life_uncertainty_seconds / 60
    sr_half_life_minutes_string, sr_half_life_uncertainty_minutes_string = format_value_with_uncertainty(sr_half_life_minutes,
                                                                                                         sr_half_life_uncertainty_minutes, 3)

    rb_half_life_minutes = rb_half_life_seconds / 60
    rb_half_life_uncertainty_minutes = rb_half_life_uncertainty_seconds / 60
    rb_half_life_minutes_string, rb_half_life_uncertainty_minutes_string = format_value_with_uncertainty(rb_half_life_minutes,
                                                                                                         rb_half_life_uncertainty_minutes, 3)

    figure = plt.figure(figsize=(10, 9))
    figure_grid = gridspec.GridSpec(2, 1, height_ratios=[6, 1])

    main_plot = figure.add_subplot(figure_grid[0])
    main_plot.plot(xs, calculate_activity_rb(xs, sr_decay_constant, rb_decay_constant))
    main_plot.errorbar(times, activities, yerr=activities_error, fmt='kx', capsize=2, elinewidth=0.75)

    main_plot.set_title('Activity of the Strontium sample against time', fontsize=16)
    main_plot.set_xlabel('Time (seconds)')
    main_plot.set_ylabel('Activity (TBq)')
    main_plot.annotate('Reduced $\chi^2$ = {:.2f}'.format(reduced_chi2), (0, 0), (0, -40),
                       xycoords='axes fraction', textcoords='offset points',
                       va='top')
    main_plot.annotate(
        r'$\lambda_{Rb}$ = ' + '{} ± {}'.format(rb_decay_constant_string, rb_decay_constant_uncertainty_string)
        + r' s$^{-1}$', (0, 0), (0, -60),
        xycoords='axes fraction', textcoords='offset points',
        va='top')
    main_plot.annotate(
        r'Rb: $t_{1/2}$ = ' + '{} ± {} minutes'.format(rb_half_life_minutes_string, rb_half_life_uncertainty_minutes_string),
        (0, 0), (200, -60),
        xycoords='axes fraction', textcoords='offset points',
        va='top')
    main_plot.annotate(
        r'$\lambda_{Sr}$ = ' + '{} ± {}'.format(sr_decay_constant_string, sr_decay_constant_uncertainty_string)
        + r' s$^{-1}$', (0, 0), (0, -80),
        xycoords='axes fraction', textcoords='offset points',
        va='top')
    main_plot.annotate(
        r'Sr:  $t_{1/2}$ = ' + '{} ± {} minutes'.format(sr_half_life_minutes_string, sr_half_life_uncertainty_minutes_string),
        (0, 0), (200, -80),
        xycoords='axes fraction', textcoords='offset points',
        va='top')

    legend_strings = [
        r'Minimised $\chi^2$ fit of: A$_{Rb}(t) = \lambda_{Rb} N_{Sr}(0) \frac{\lambda_{Rs}}{\lambda_{Rb} - \lambda_{Sr}} \left[ \exp(-\lambda_{Sr}t) - \exp(-\lambda_{Sr}t)\right] $',
        'Data'
    ]
    main_plot.legend(legend_strings)

    residual_plot = figure.add_subplot(figure_grid[1])
    residual_plot.errorbar(times, residuals, activities_error, fmt='kx', capsize=2, elinewidth=0.75)
    residual_plot.plot([min(times), max(times)], [0, 0])

    residual_plot.set_title('Residuals in Activity')
    residual_plot.set_xlabel('Time (seconds)')
    residual_plot.set_ylabel('Activity (TBq)')

    figure.tight_layout()
    figure.savefig(main_plot_save_filepath)
    figure.show()

    print_final_values(sr_decay_constant_string, sr_decay_constant_uncertainty_string,
                       rb_decay_constant_string, rb_decay_constant_uncertainty_string,
                       sr_half_life_minutes_string, sr_half_life_uncertainty_minutes_string,
                       rb_half_life_minutes_string, rb_half_life_uncertainty_minutes_string)


def plot_chi_squared_contour(times, activities, activity_errors, sr_decay_constant, rb_decay_constant):
    """
    Plots the chi squared contour plot with specified chi squared valjues marked on.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    contour_plot : QuadContourSet
    """
    print_separator('Plotting chi squared contour graph.')
    # Calculate rage of possible decay constant value.
    sr_decay_constant_lower = sr_decay_constant - 0.0002
    sr_decay_constant_upper = sr_decay_constant + 0.0002

    rb_decay_constant_lower = rb_decay_constant - 0.00001
    rb_decay_constant_upper = rb_decay_constant + 0.00001

    # Generate 1D Array for possible decay constant values.
    sr_decay_constant_values = np.linspace(sr_decay_constant_lower, sr_decay_constant_upper, 100)
    rb_decay_constant_values = np.linspace(rb_decay_constant_lower, rb_decay_constant_upper, 100)

    # Mesh the 1D Arrays of possible decay constant values.
    sr_decay_constants_mesh = mesh_array(sr_decay_constant_values)
    rb_decay_constants_mesh = mesh_array(rb_decay_constant_values)

    # Take the transpose of rb decay constants.
    rb_decay_constants_mesh = np.transpose(rb_decay_constants_mesh)

    # Calculate Chi squared values.
    chi_squared_values = calculate_chi_2(times, activities, activity_errors,
                                         sr_decay_constants_mesh,
                                         rb_decay_constants_mesh)

    minimised_chi_squared = calculate_chi_2(times, activities, activity_errors, sr_decay_constant, rb_decay_constant)

    # Contour values
    contour_values = [minimised_chi_squared]
    for difference in key_chi_squared_differences:
        chi_squared = minimised_chi_squared + difference
        contour_values.append(chi_squared)

    figure = plt.figure(figsize=(7, 5))

    chi_squared_plot = figure.add_subplot(111)

    chi_squared_plot.set_title('Plot of $\chi^2$ for various $\lambda_{Rb}$ and $\lambda_{Sr}$')
    chi_squared_plot.set_xlabel(r'$\lambda_{Sr}$')
    chi_squared_plot.set_ylabel(r'$\lambda_{Rb}$')

    chi_squared_plot.contourf(sr_decay_constants_mesh, rb_decay_constants_mesh, chi_squared_values, cmap='Blues')

    contour_plot = chi_squared_plot.contour(sr_decay_constants_mesh, rb_decay_constants_mesh, chi_squared_values,
                                    contour_values, linestyles = 'dashed', colors = 'k')

    chi_squared_plot.clabel(contour_plot)

    # Minimized chi squared point.
    minimum_point = chi_squared_plot.scatter(sr_decay_constant, rb_decay_constant)
    minimum_point.set_label(r'Min $\chi^2$ = {:.1f}'.format(minimised_chi_squared))

    # Set Axis limits
    chi_squared_plot.set_xlim((sr_decay_constant_lower, sr_decay_constant_upper))
    chi_squared_plot.set_ylim((rb_decay_constant_lower, rb_decay_constant_upper))

    # Modify x-axis ticks
    start_x, end_x = chi_squared_plot.get_xlim()
    x_increment = (end_x - start_x) / 4
    x_ticks = np.arange(start_x, end_x, x_increment)
    chi_squared_plot.set_xticks(x_ticks)

    # Legend
    legend_labels = ['Min']
    for difference in key_chi_squared_differences:
        legend_string = r'$\chi^2$ + {:.0f} = {:.1f}'.format(difference, minimised_chi_squared + difference)
        legend_labels.append(legend_string)

    for index in range(1, len(legend_labels) - 1, 1):
        chi_squared_plot.collections[index].set_label(legend_labels[index])
    chi_squared_plot.legend(loc='center left', bbox_to_anchor=(1, 0.8), fontsize=10)

    figure.tight_layout()
    figure.savefig(contour_plot_save_filepath, ppi=300)
    figure.show()

    return contour_plot


def plot_3d_contour(sr_):
    figure = plt.figure(figsize=(7, 5))
    chi_squared_plot = figure.add_subplot(111)
    chi_squared_plot.contour3D()

# MARK: - Plotting calculations.
def calculate_residuals(times, activities, sr_decay_constant, rb_decay_constant):
    """
    Calculates the residuals in activity.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activities_error : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    residuals : numpy.ndarray
    """
    residuals = activities - calculate_activity_rb(times, sr_decay_constant, rb_decay_constant)
    return residuals


def calculate_chi_2(times, activities, activities_errors, sr_decay_constant, rb_decay_constant):
    """
    Calculates the chi squared
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activities_errors : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    chi2: float
    """
    chi2 = 0
    for i in range(len(times)):
        difference = activities[i] - calculate_activity_rb(times[i], sr_decay_constant, rb_decay_constant)
        error = activities_errors[i]
        chi2 += pow(difference / error, 2)

    return chi2


def calculate_reduced_chi_2(times, activities, activities_errors, sr_decay_constant, rb_decay_constant):
    """
    Calculates the reduced chi squared
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activities_errors : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    reduced_chi: float
    """
    chi2 = calculate_chi_2(times, activities, activities_errors, sr_decay_constant, rb_decay_constant)
    degrees_of_freedom = len(times) - 2
    reduced_chi2 = chi2 / degrees_of_freedom
    print('Reduced chi2 = {}'.format(reduced_chi2))
    return reduced_chi2


def get_chi_squared_function(times, activities, activity_errors):
    """
    Sets up and returns the chi squared function so that it can be minimised.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    Returns
    --------
    transmission_coefficients : np.array([float])
    energies : np.array([float])
    transmission_coefficient_errors : np.array([float])
    """
    return lambda decay_constants: calculate_chi_2(times, activities, activity_errors, decay_constants[0],
                                                   decay_constants[1])


def calculate_minimum_fit(times, activities, activity_errors, sr_decay_constant, rb_decay_constant):
    """
    Calculates the fit which minimises chi squared.
    Parameters
    --------
    times : numpy.ndarray
    activities : numpy.ndarray
    activity_errors : numpy.ndarray
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    transmission_coefficients : np.array([float])
    energies : np.array([float])
    transmission_coefficient_errors : np.array([float])
    """
    print('Calculating minimum chi squared fit.')
    chi_squared_function = get_chi_squared_function(times, activities, activity_errors)

    sr_decay_constant, rb_decay_constant = fmin(chi_squared_function, (sr_decay_constant, rb_decay_constant))

    sr_decay_constant_string = format_to_significant_figures(sr_decay_constant, 3)
    rb_decay_constant_string = format_to_significant_figures(rb_decay_constant, 3)

    print('sr_decay_constant = {} per second'.format(sr_decay_constant_string))
    print('rb_decay_constant = {} per second'.format(rb_decay_constant_string))

    return sr_decay_constant, rb_decay_constant


def calculate_uncertainty_from_contour_plot(contours):
    """
    Calculates the un-correlated errors using half the max width and height of the width.
    Parameters
    --------
    contours : QuadContourSet
    Returns
    --------
    x_error : float
    y_error : float
    """
    contour_path = contours.allsegs[1][0]

    xs = contour_path[:, 0]
    ys = contour_path[:, 1]

    min_x = min(xs)
    max_x = max(xs)

    min_y = min(ys)
    max_y = max(ys)

    x_error = (max_x - min_x) / 2
    y_error = (max_y - min_y) / 2

    print('x error = {:.4g}'.format(x_error))
    print('y error = {:.4g}'.format(y_error))
    return x_error, y_error


def mesh_array(array):
    """
    Mesh the given array. (Mesh = form a square 2D array of the original 1D array)
    Parameters
    --------
    array : numpy.ndarray
    Returns
    --------
    meshed_array : numpy.ndarray
    """
    meshed_array = np.empty((0, len(array)))

    for index in range(len(array)):
        meshed_array = np.vstack((meshed_array, array))

    return meshed_array


# MARK: - Activity calculations.
def calculate_activity_rb(time, sr_decay_constant, rb_decay_constant):
    """
    Calculates the activity of Rb for a given time using the minimized fit.
    Parameters
    --------
    time : float
    sr_decay_constant : float
    rb_decay_constant : float
    Returns
    --------
    rb_activity : float
    """
    # if (rb_decay_constant - sr_decay_constant == 0:
    #     return np.inf
    return rb_decay_constant * number_of_molecules_in_sample * sr_decay_constant * (1 / (rb_decay_constant - sr_decay_constant)) \
           * (pow(np.e, - sr_decay_constant * time) - pow(np.e, - rb_decay_constant * time)) * pow(10, -12)


def calculated_half_life(decay_constant, decay_constant_uncertainty):
    """
    Calculates the half-life in seconds.
    Parameters
    --------
    decay_constant : float
    decay_constant_uncertainty : float
    Returns
    --------
    half_life_seconds : float
    half_life_uncertainty_seconds : float
    """
    half_life_seconds = np.log(2) / decay_constant
    half_life_uncertainty_seconds = (decay_constant_uncertainty / decay_constant) * half_life_seconds

    return half_life_seconds, half_life_uncertainty_seconds


# MARK: - String Formatting
def format_to_significant_figures(value, significant_figures):
    """
    Formats the given value to the specified significant figures.
    Parameters
    --------
    value : float
    significant_figures : int
    Returns
    --------
    value_rounded_string : str
    """
    format_string = '{:.' + str(significant_figures) + 'g}'
    value_rounded_string = format_string.format(value)

    start_index = len(value_rounded_string) - 1
    end_index = start_index
    for index in range(len(value_rounded_string) - 1, -1, -1):
        if not value_rounded_string[index] == '0' and not value_rounded_string[index] == '.':
            end_index = index

    current_significant_figures = start_index - end_index + 1

    while current_significant_figures < significant_figures:
        value_rounded_string += '0'
        current_significant_figures += 1

    return value_rounded_string


def format_value_with_uncertainty(value, uncertainty, significant_figures):
    """
    Formats the given value and uncertainty to the specified significant figures
    with the uncertainty having the same number of decimal places.
    Parameters
    --------
    value : float
    uncertainty : float
    significant_figures : int
    Returns
    --------
    formatted_value : str
    formatted_uncertainty : str
    """
    formatted_value = format_to_significant_figures(value, significant_figures)

    # Count value decimal places
    decimal_points = 0
    for index in range(len(formatted_value) - 1, -1, -1):
        if not formatted_value[index] == '.':
            decimal_points += 1
        else:
            break

    format_string = '{:.' + str(decimal_points) + 'f}'
    formatted_uncertainty = format_string.format(uncertainty)

    return formatted_value, formatted_uncertainty


# MARK: - Utils
def print_separator(label):
    print('***** {} ******'.format(label))


def print_final_values(sr_decay_constant, sr_decay_constant_uncertainty, rb_decay_constant, rb_decay_constant_uncertainty,
                       sr_half_life, sr_half_life_uncertainty, rb_half_life, rb_half_life_uncertainty):
    print('79-Sr decay constant = {} ± {} per second'.format(sr_decay_constant, sr_decay_constant_uncertainty))
    print('79-Rb decay constant = {} ± {} per second'.format(rb_decay_constant, rb_decay_constant_uncertainty))
    print('79-Sr half-life = {} ± {} minutes'.format(sr_half_life, sr_half_life_uncertainty))
    print('79-Rb half-life = {} ± {} minutes'.format(rb_half_life, rb_half_life_uncertainty))


if __name__ == '__main__':
    main()
