import random
import numpy as np
import torch



def flat_accuracy(preds, labels):
    _,pred_flat = torch.max(preds.data, 1)
    labels_flat = labels.data
    return (pred_flat == labels_flat).sum().item() / len(labels_flat)


def train(model,train_dataloader,validation_dataloader,criterion,optimizer,epochs=4,device = torch.device("cuda")):
    # Set the seed value all ovr the place to make this reproducible.
    seed_val = 2

    random.seed(seed_val)
    torch.manual_seed(seed_val)
    
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')


        total_loss = 0
        total_accuracy=0


        model.train()
        total_loss = 0
        for step, (inputs,labels) in enumerate(train_dataloader):

            if step % 20 == 0 and not step == 0:

                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

 
            model.zero_grad()        
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs,labels)


            total_loss += loss.item()

            loss.backward()


            optimizer.step()


        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)  
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))        
        
        # ========================================
        #               Validation
        # ========================================


        print("")
        print("Running Validation...")

        model.eval()
        eval_accuracy = 0
        eval_loss = 0

        for step, (inputs,labels) in enumerate(validation_dataloader):

            inputs, labels = inputs.to(device),labels.to(device)

            with torch.no_grad():        


                outputs = model(inputs)
                tmp_eval_loss = criterion(outputs,labels)

            # Calculate the accuracy for this batch of test .
            tmp_eval_accuracy = flat_accuracy(outputs, labels)
            # Accumulate the total sc_ore.
            eval_accuracy += tmp_eval_accuracy
            eval_loss += tmp_eval_loss

        print("Average Validation Accuracy: {0:.2f}".format(eval_accuracy/len(validation_dataloader)))
        print("Average Validation  Loss: {0:.2f}".format(eval_loss/len(validation_dataloader)))

        
        

def test(model, test_dataloader,device= torch.device("cuda")):
    # ========================================
    #               Test
    # ========================================


    print("")
    print("Running Test...")


  
    model.eval()
    eval_accuracy = 0
    # Evaluate data for one epoch
    for batch in test_dataloader:

        inputs, labels = batch[0].to(device),batch[1].to(device)

        with torch.no_grad():        


            outputs = model(inputs)

        # Calculate the accuracy for this batch of test .
        tmp_eval_accuracy = flat_accuracy(outputs, labels)
        # Accumulate the total score.
        eval_accuracy += tmp_eval_accuracy

    print("  Accuracy: {0:.2f}".format(eval_accuracy/len(test_dataloader)))