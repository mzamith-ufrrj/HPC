#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <CellularAutomata.cuh>
#include <App.h>

using namespace std;

Stopwatch stopwatch;
CellularAutomata *cellularAutomata;
double elapsedTime = 0.0f;
float FPS = 0.0f;
int width = 800,
    height = 600;
bool start = false;
// Função de visualização
void render(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    

    cellularAutomata->render();

    float delta  = 0.0f;
    glColor3f(0.0f, 0.0f, 1.0f);
//    for (int i = 0; i < cellularAutomata->getCellX(); i++)

    for (int j = 0; j <= cellularAutomata->getCellY(); j++){
        glBegin(GL_LINES); //GL_QUADS);GL_LINE_LOOP
        glVertex3f(0.0f, -delta, 0.0f);
        glVertex3f(1.0f, -delta, 0.0f);
        glEnd();
        delta += cellularAutomata->getScaleY();
    }

    delta = 0.0f;
    for (int j = 0; j <= cellularAutomata->getCellX(); j++){
        glBegin(GL_LINES); //GL_QUADS);GL_LINE_LOOP
        glVertex3f(delta, 0.0f, 0.0f);
        glVertex3f(delta, -1.0f, 0.0f);
        glEnd();
        delta += cellularAutomata->getScaleX();
    }
    
    
    glutSwapBuffers();
    

}

// Função de inicialização de parâmetros (OpenGL e outros)
void init (void)
{ 
   glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

// Função de evento do teclado
void keyboardEvent(unsigned char key, int x, int y)
{
     //    glutPostRedisplay();
    switch (key) {
        case 'b':
        case 'B':
            start = true;
            break;

        case 'e':
        case 'E':
             start = false;
            break;
        case 'c':
        case 'C':
            cellularAutomata->clear();
            break;
     
        case 'r':
        case 'R':
            //cellularAutomata->random();
            break;

        case 's':
        case 'S':
            cellularAutomata->update(0.0f);
            break;

        case 'q':
        case 'Q':
        case 27:
            exit (EXIT_SUCCESS);
            break;
            
    
        case 't':
            //cellularAutomata->decreaseTime();
            break;
        case 'T':
            //cellularAutomata->increaseTime();
            break;
        case 'd':
            //cellularAutomata->decreaseDistance();
            break;
        case 'D':
            //cellularAutomata->increaseDistance();
            break;
            
        default:
            break;
   }
}

// Função de evento do mouse
void mouseEvent(int button, int state, int x, int y)
{
   /*make
    if (button == GLUT_LEFT_BUTTON)
        if (state == GLUT_DOWN){
            float x1 = (static_cast<float>(x) / width) * static_cast<float>(cellularAutomata->getCellX());
            float y1 = (static_cast<float>(y) / height) * static_cast<float>(cellularAutomata->getCellY());
            int x2 = static_cast<int>(x1)  ;
            int y2 = static_cast<int>(y1) ;
            //x2 /=  cellularAutomata->getCellX();
            //y2 /=  cellularAutomata->getCellY();

            cellularAutomata->changeState(x2, y2);
            
        }
        */
    
    if (button == GLUT_RIGHT_BUTTON)
        if (state == GLUT_DOWN)
            cerr << "Right button" << endl;
    
   glutPostRedisplay();
}

//Viewport
void viewPort(int w, int h)
{
    
    if(h == 0) h = 1;
    
    
    width = w;
    height = h;
    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho (-0.01f, 1.01f, -1.01f, 0.01f, -1.0f, 1.0f);
    glutPostRedisplay();
}

//Loop principal da visualização
void mainloop(void)
{
   

    
//    cellularAutomata->update();
    glutPostRedisplay();
    STOP_STOPWATCH(stopwatch);
    elapsedTime += stopwatch.mElapsedTime;

    if (start)
        cellularAutomata->update(stopwatch.mElapsedTime);

    FPS++;
    if (FPS >= 100.0f){
        float realfps = 1.0f / (elapsedTime / 100.0f);
        char msg[1024];
//        sprintf(msg, "CA - Game of life \t \t Alive: %.2f \t Dead: %.2f \t FPS: %5.2f", cellularAutomata->getAlive(), cellularAutomata->getDead(), realfps);
        sprintf(msg, "CA - Game of life \t \t Alive: %.2f \t Dead: %.2f", cellularAutomata->getAlive(), cellularAutomata->getDead());
        glutSetWindowTitle(msg);
        FPS == 0.0f;
        elapsedTime = 0.0f;
    }
    START_STOPWATCH(stopwatch);
   
}


int main(int argc, char**argv)
{
    int x = 8,
        y = 8;
    

    if (argc == 3){
        x = atoi(argv[1]);
        y = atoi(argv[2]);
    }
    
    cellularAutomata = new CellularAutomata(x, y);
    START_STOPWATCH(stopwatch);
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(width, height);
    glutCreateWindow("Cellular Automata - Game of life");
    glutDisplayFunc(render);
    glutReshapeFunc(viewPort);
    glutMouseFunc(mouseEvent);
    glutKeyboardFunc(keyboardEvent);
    glutIdleFunc(mainloop);
    init();
    glutMainLoop();
    return EXIT_SUCCESS;
}
