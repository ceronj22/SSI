
#7/12/21

import csv


#for reference
header = [["Time", "Velocity", "Wheel Angle", "Push Button", "X Pose", "Y Pose", "Z Orien", "X PF", "Y PF", "Z Orien PF"]]


#2D list to store all the values in
csv_data = []

app_folder = 'Data_Collection/'



def flip_data():
    
    #cut the header off to more easily manipulate the numbers within
    del csv_data[0]
    
    
    
    for row in csv_data:
        #flip wheel angles
        row[2] = 1 - float(row[2]) #= -(wheel_angle - 0.5) + 0.5
        
        #my left and right is y and my forward back is x. should be changed in another script,
        #but I figure for now I should just flip my data appropriately
        row[8] = float(row[8]) * -1
        
        #swap Zorien - I think this can be flipped by making Z orien negative?
        row[9] = float(row[9]) * -1
    
    
    #add the header back to the front
    csv_data.insert(0, header)



#takes in the name of the csv to read and stores its data to csv_data 2D list
def read_csv(csv_name):
    #clear any data csv_data has in it already
    del csv_data[:]
    
    csv_name += ".csv"

    with open(str(app_folder + csv_name)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        #add every row of the read csv to the csv_data 2d list
        for row in csv_reader:
            csv_data.append(row)
    
    #print(csv_data)



#takes collected data and writes it to the csv file in the same directory
def write_to_csv(csv_name):
    
    #open the file - cleaner than having to close seperately
    with open(str(app_folder + csv_name + ".csv"), 'w+') as file:
        #create a csv writer
        writer = csv.writer(file)
        
        #print('csv_data: {}'.format(csv_data))
        
        #write all rows to that csv file
        writer.writerows(csv_data)
        print("Data saved to csv {}!".format(csv_name))


#helper function to iterate through all of the csvs
def flip_csvs():
    
    #iterate through all shoebox and tissue box positions
    for sb_pos in range(7):
        for tb_pos in range(5):
            run = "{}-{}".format(sb_pos, tb_pos)
            
            #read the csv of the given run -> over
            read_csv(run)
            
            #flip that data accordingly
            flip_data()
            
            #write that data to an accordingly named .csv file
            write_to_csv(str("flipped_" + run))
            
            



if __name__ == '__main__':
    
    print("Running main!")
    flip_csvs()
    print("All .csvs iterated through")

#STEPS NEEDED:
#Pose:
    #Flip sign on y values
    #(swap x and y) (another script)
#flip wheel angles:
    #-(wheel_angle - 0.5) + 0.5 = 1 - wheel_angle
#Zorien:
    #figure this one out... flip sign?
    #1 --> 180
    # i think flipping sign is good?

#PF:
    #align PF with other vals? (pf lags so)
