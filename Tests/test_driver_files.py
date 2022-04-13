import pandas
import os


def get_driving_events(file):
    # reading the CSV file
    csvFile = pandas.read_csv(file)
    # displaying the contents of the CSV file
    print(csvFile, '\n')


def get_mobileye_advanced_warning_system(file):
    # reading the CSV file
    csvFile = pandas.read_csv(file)
    # displaying the contents of the CSV file
    print(csvFile, '\n')


def get_mobileye_traffic_sign_recognition(file):
    # reading the CSV file
    csvFile = pandas.read_csv(file)
    # displaying the contents of the CSV file
    print(csvFile, '\n')


def get_mobileye_car_information(file):
    # reading the CSV file
    csvFile = pandas.read_csv(file)
    # displaying the contents of the CSV file
    print(csvFile, '\n')


def get_gps(file):
    # reading the CSV file
    csvFile = pandas.read_csv(file)
    # displaying the contents of the CSV file
    print(csvFile, '\n')


if __name__ == "__main__":

    # get driving events - Useful
    # events_file = 'files/trips/30480_2021_08_01T08_27_02-DrivingEvents_Map.csv'
    # get_driving_events(events_file)

    # Gives information about the safety and warning state of the Mobileye
    # system - Not very useful
    # events_file = 'files/trips/30480_2021_08_01T08_27_02-ME_AWS.csv'
    # get_mobileye_advanced_warning_system(events_file)

    # Gives information about the detected road traffic signs by Mobileye
    # - Useful
    # TODO: Questoes
    # - Os valores do tsr_x_sup são o que?
    # events_file = 'files/trips/30480_2021_08_01T08_27_02-ME_TSR.csv'
    # get_mobileye_traffic_sign_recognition(events_file)

    # Gives information about the about the car parameters needed for the
    # Mobileye system
    # - Useful
    # TODO: Questoes
    # - O speed é o speed atual do veiculo?
    # - Os valores são obtidos segundo a segundo?
    # events_file = 'files/trips/30480_2021_08_01T08_27_02-ME_Car.csv'
    # get_mobileye_car_information(events_file)

    # Satellite-based geolocation data, about one sample per second
    # - Useful
    # TODO: Questoes
    # - São os dados enviados de segundo a segundo?
    # - O que é o heading-direção? (hdg) e a altitude (alt)
    # events_file = 'files/trips/30480_2021_08_01T08_27_02-GPS.csv'
    # get_gps(events_file)

    # test if trips all had the same events
    # Conclusion: Trips have diferent events, which mean they will have 
    # different files. Some trips may have more events than others which will
    # result on having more files

    path_of_the_directory = "D:/Mestrado/Tese/workspace/Tests/trips"
    for root, subdirectories, files in os.walk(path_of_the_directory):
        for subdirectory in subdirectories:
            print(subdirectory)
            directory = os.listdir(os.path.join(root, subdirectory))
            for file in directory:
                print(file, ' ', end="")
            print('\n')
