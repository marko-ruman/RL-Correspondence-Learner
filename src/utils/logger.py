import os
import pickle


class Logger:
    def __init__(self, dir_score="score/", dir_memory="memory/", dir_results="results/"):
        self.dir_score = dir_score
        self.dir_memory = dir_memory
        self.dir_results = dir_results

    @staticmethod
    def _write_to_file(number, file, dir_):
        try:
            number = number.cpu().detach().numpy()
        except:
            pass
        if not os.path.isdir(dir_):
            os.makedirs(dir_)
        f = open(dir_+"/"+file+".csv", "a+")
        f.write(";" + str(number))
        f.close()

    def save_errors(self, errors):

        for error_name, error in errors.items():
            self._write_to_file(error, error_name, self.dir_results)

        if "a_dis_loss" in errors and "b_dis_loss" in errors:
            self._write_to_file(errors["a_dis_loss"]+errors["b_dis_loss"], "dis_loss", self.dir_results)

        if "a_gen_loss" in errors and "b_gen_loss" in errors:
            self._write_to_file(errors["a_gen_loss"] + errors["b_gen_loss"], "gen_loss", self.dir_results)

        if "a_cycle_loss" in errors and "b_cycle_loss" in errors:
            self._write_to_file(errors["a_cycle_loss"]+errors["b_cycle_loss"], "cycle_loss", self.dir_results)

        if "a_q_loss" in errors and "b_q_loss" in errors:
            self._write_to_file(errors["a_q_loss"]+errors["b_q_loss"], "q_loss", self.dir_results)

        if "a_q_loss" in errors and "b_q_loss" in errors and "q_cycle_loss" in errors:
            self._write_to_file(errors["a_q_loss"] + errors["b_q_loss"] + errors["q_cycle_loss"], "full_q_loss",
                                self.dir_results)

        try:
            self._write_to_file(errors["a_gen_loss"] + errors["b_gen_loss"] + errors["a_cycle_loss"]
                                + errors["b_cycle_loss"] + errors["a_q_loss"] + errors["a_model_loss"], "full_gen_loss",
                                self.dir_results)
        except:
            pass

    def save_avg_score(self, score):
        self._write_to_file(score, "avg_score", self.dir_results)

    def save_replay_memory(self, replay_memory, filename):
        data = []
        for i, m in enumerate(replay_memory.state_memory):
            data.append((m, replay_memory.action_memory[i], replay_memory.new_state_memory[i], replay_memory.reward_memory[i]))
        pickle.dump(data, open(self.dir_memory + "/"+ filename, "wb+"), protocol=4)
        # pickle.dump(replay_memory, open(filename, "wb+"), protocol=4)


