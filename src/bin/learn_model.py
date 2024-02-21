import os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import pickle
from src.models.environment_model import EnvironmentModel


def square_concat(image):
    image = image[0].cpu().detach().numpy()
    c = np.concatenate(image[:int(image.shape[0]/2)], axis=1)
    d = np.concatenate([c, np.concatenate(image[int(image.shape[0]/2):], axis=1)])
    d.resize((1,1) + d.shape)
    d = torch.tensor(d)
    return d


def save_image(state, result, out, out_a, epoch, i):
    # generate the batch image
    state = square_concat(state)
    result = square_concat(result)
    out = square_concat(out)
    diff = result-out
    merge_img = []
    for (before, after, pred, d) in zip(state, result, out, diff):
        img = torch.cat([before, after, pred, d], 1)
        merge_img.append(img)

    merge_img = torch.cat(merge_img, 2).cpu()
    merge_img = (merge_img + 1) / 2
    img = transforms.ToPILImage()(merge_img)
    # img = transforms.Resize((512, 640))(img)
    img.save(os.path.join('./log/env_model_img_log', 'img_{}_{}.jpg'.format(epoch, i)))

    merge_img = []
    out_a_cat = []
    for j, o in enumerate(out_a):
        obr = square_concat(o)[0]
        if j > 0:
            obr = obr - out_a_cat[0]
        out_a_cat.append(obr)
    img = torch.cat(out_a_cat, 1)
    merge_img.append(img)

    merge_img = torch.cat(merge_img, 2).cpu()
    merge_img = (merge_img + 1) / 2
    img = transforms.ToPILImage()(merge_img)
    img.save(os.path.join('./log/env_model_img_log', 'action_img_{}_{}.jpg'.format(epoch, i)))


def train_environment_model(dataset_filename, opt):
    model = EnvironmentModel(opt).cuda()

    dataset = pickle.load(open('memory/' + dataset_filename, "rb"))

    batch_size = 16
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        for i, item in enumerate(loader):

            # each item contains state, action, new_state and reward
            state,action_disc,result,reward = item

            action = torch.zeros((batch_size, opt["action_dim"]))
            for index, a in enumerate(action_disc):
                action[index, a] = 1
            state = state.float().cuda()*2.0-1
            action = action.float().cuda()
            result = result.float().cuda()*2.0-1
            try:
                out = model(state, action)
                loss = loss_fn(out, result)*100
                print(epoch, i, loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except:
                print("ERROR")
            if i%100 == 0:
                out_a = []
                for a in range(opt["action_dim"]):
                    action = torch.zeros(opt["action_dim"])
                    action[a] = 1
                    out_a.append(model(state[:1], action.float().cuda()))
                save_image(state, result, out, out_a, epoch, i)
        torch.save(model.state_dict(),'./imgpred.pth')



opt = {
    "action_dim": 6
}

train_environment_model("pong", opt)