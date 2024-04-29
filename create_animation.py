from functions import *


def do_the_animation(info, to_save=False):
    img_np = info['img_np']
    agents = info['agents']
    max_time = info['max_time']
    img_dir = info['img_dir']
    alg_name = info['alg_name']
    n_agents = len(agents)
    i_agent = agents[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    field = img_np * -1
    ax[0].imshow(field, origin='lower')
    ax[0].set_title(f'{n_agents} agents, {max_time} steps')

    others_y_list, others_x_list, others_cm_list = [], [], []
    for agent in agents:
        curr_node = agent.start_node
        others_y_list.append(curr_node.y)
        others_x_list.append(curr_node.x)
        others_cm_list.append(get_color(agent.num))
    scat1 = ax[0].scatter(others_y_list, others_x_list, s=100, c='k')
    scat2 = ax[0].scatter(others_y_list, others_x_list, s=50, c=np.array(others_cm_list))

    goal_scat1 = ax[0].scatter([i_agent.goals_per_iter_list[0].y], [i_agent.goals_per_iter_list[0].x], s=200, c='white', marker='X')
    goal_scat2 = ax[0].scatter([i_agent.goals_per_iter_list[0].y], [i_agent.goals_per_iter_list[0].x], s=100, c='red', marker='X')

    agent_scat1 = ax[0].scatter([i_agent.start_node.y], [i_agent.start_node.x], s=120, c='w')
    agent_scat2 = ax[0].scatter([i_agent.start_node.y], [i_agent.start_node.x], s=70, c='r')

    def update(frame):
        # for each frame, update the data stored on each artist.
        fr_y_list, fr_x_list = [], []
        for agent in agents:
            fr_node = agent.path[frame]
            fr_y_list.append(fr_node.y)
            fr_x_list.append(fr_node.x)
        # update the scatter plot:
        data = np.stack([fr_y_list, fr_x_list]).T
        scat1.set_offsets(data)
        scat2.set_offsets(data)

        fr_i_node = i_agent.path[frame]
        data = np.stack([[fr_i_node.y], [fr_i_node.x]]).T
        agent_scat1.set_offsets(data)
        agent_scat2.set_offsets(data)

        fr_i_goal = i_agent.goals_per_iter_list[frame]
        data = np.stack([[fr_i_goal.y], [fr_i_goal.x]]).T
        goal_scat1.set_offsets(data)
        goal_scat2.set_offsets(data)

        return scat1, scat2, agent_scat1, agent_scat2, goal_scat1, goal_scat2

    ani = animation.FuncAnimation(fig=fig, func=update, frames=max_time, interval=250)
    if to_save:
        add_text = f'{alg_name}_'
        ani.save(filename=f"../videos/{add_text}{n_agents}_agents_in_{img_dir[:-4]}_for_{max_time}_steps.mp4", writer="ffmpeg")
    plt.show()

