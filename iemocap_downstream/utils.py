import torch

def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    train_loss = 0
    for batch in train_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)

        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(feats, speech_padding_mask)
        
        loss = criterion(outputs, labels.long())

        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()

    return train_loss

@torch.no_grad()
def validate_and_test(model, data_loader, device, num_classes):
    model.eval()
    correct, total = 0, 0

    # unweighted accuracy
    unweightet_correct = [0] * num_classes
    unweightet_total = [0] * num_classes

    # weighted f1
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for batch in data_loader:
        ids, net_input, labels = batch["id"], batch["net_input"], batch["labels"]
        feats = net_input["feats"]
        speech_padding_mask = net_input["padding_mask"]

        feats = feats.to(device)

        speech_padding_mask = speech_padding_mask.to(device)
        labels = labels.to(device)

        outputs = model(feats, speech_padding_mask)

        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
        for i in range(len(labels)):
            unweightet_total[labels[i]] += 1
            if predicted[i] == labels[i]:
                unweightet_correct[labels[i]] += 1
                tp[labels[i]] += 1
            else:
                fp[predicted[i]] += 1
                fn[labels[i]] += 1

    weighted_acc = correct / total * 100
    unweighted_acc = compute_unweighted_accuracy(unweightet_correct, unweightet_total) * 100
    weighted_f1 = compute_weighted_f1(tp, fp, fn, unweightet_total) * 100

    return weighted_acc, unweighted_acc, weighted_f1

def inference(model, ):
    pass


def compute_unweighted_accuracy(list1, list2):
    result = []
    for i in range(len(list1)):
        result.append(list1[i] / list2[i])
    return sum(result)/len(result)

def compute_weighted_f1(tp, fp, fn, unweightet_total):
    f1_scores = []
    num_classes = len(tp)
    
    for i in range(num_classes):
        if tp[i] + fp[i] == 0:
            precision = 0
        else:
            precision = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] == 0:
            recall = 0
        else:
            recall = tp[i] / (tp[i] + fn[i])
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))
            
    wf1 = sum([f1_scores[i] * unweightet_total[i] for i in range(num_classes)]) / sum(unweightet_total)
    return wf1
