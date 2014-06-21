from Image_Processing import *
from solve_sudoku import *


def main(image):

    board,position,final_image = image_processing(image)
    start = time.clock()

    
    solved_sudoku = solve(board)
    #print solved_sudoku
    t = time.clock()-start
    print t

    if(solved_sudoku):
        output=[]
        for s in squares:
            output.append(solved_sudoku[s])
        


        #print output
        for i in range(len(position)):
            if(position[i] !=0):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_image,output[i],position[i], font,1.1,(0,255,0),2,cv2.CV_AA)

        output = final_image.copy()
        cv2.imwrite('output.jpg',output)
        cv2.imshow('output',final_image)
        cv2.waitKey(0)
        return True
        

    else:
        print 'not solved!'
        return False

main('sudoku.jpg')

