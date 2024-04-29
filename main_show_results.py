from globals import *
from tools_for_plotting import *


def show_results(file_dir: str, lmapf: bool = False) -> None:
    plt.close()
    with open(f'{file_dir}', 'r') as openfile:
        # Reading from json file
        logs_dict = json.load(openfile)

        if lmapf:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            plot_throughput(ax[0], logs_dict)
            plot_en_metric_cactus(ax[1], logs_dict)

        else:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_sr(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_sq_metric_cactus(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_en_metric_cactus(ax, logs_dict)
            plt.show()

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_time_metric_cactus(ax, logs_dict)
            plt.show()

        plt.show()


def main():
    file_dir = ''
    show_results(file_dir=f'final_logs/sacg/{file_dir}')

    # LMAPF
    # file_dir = ''
    # show_results(file_dir=f'final_logs/lmapf/{file_dir}', lmapf=True)
    pass


if __name__ == '__main__':
    main()


