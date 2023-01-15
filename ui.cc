#include "misc.hh"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <implot.h>
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers
#include <iostream>
#include "tracked.hh"
using namespace std;




static void glfw_error_callback(int error, const char* description)
{
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int graphicsThread()
{
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

  // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
  // GL ES 2.0 + GLSL 100
  const char* glsl_version = "#version 100";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
  // GL 3.2 + GLSL 150
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char* glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
  //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

  // Create window with graphics context
  GLFWwindow* window = glfwCreateWindow(1280, 720, "Hello Deep Learning", NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO(); (void)io;
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  //    ImGui::StyleColorsLight();

  // Setup Platform/Renderer backends
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  io.Fonts->AddFontFromFileTTF("/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf", 18.0f);

  // Our state
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

  shared_ptr<HyperParameters> hp;
  // Main loop
  while (!glfwWindowShouldClose(window))
    {
      auto hp = *g_hyper;
      glfwPollEvents();

      // Start the Dear ImGui frame
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      //      ImPlot::ShowDemoWindow();
      {

        ImGui::Begin("Hello, deep learning!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::SliderInt("Batch multiple", &hp.batchMult, 1, 32);
        ImGui::SliderFloat("Learning rate", &hp.lr, 0.0f, 0.2f);
        ImGui::SliderFloat("Momentum", &hp.momentum, 0.0f, 1.0f);


        if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
          std::cout<<"Clicked!"<<std::endl;
        ImGui::SameLine();
        ImGui::Text("Have %ld active nodes", TrackedNumberImp<float>::getCount());
        ImGui::Text("Trained %d characters", g_progress.trained.load());
        ImGui::Text("Last batch of %d characters took %.1f seconds", hp.getBatchSize(), g_progress.lastTook);

        if (g_progress.losses.size()>0 && ImPlot::BeginPlot("Loss plot")) {
          //          ImPlot::SetupAxes(NULL,NULL,ImPlotAxisFlags_AutoFit|ImPlotAxisFlags_RangeFit, ImPlotAxisFlags_AutoFit|ImPlotAxisFlags_RangeFit);
          ImPlot::SetupAxes("Batch", "Loss");
          ImPlot::SetupAxisLimits(ImAxis_Y1, 2, 7);
          ImPlot::SetupAxis(ImAxis_Y2, "Correct",ImPlotAxisFlags_AuxDefault);
          ImPlot::SetupAxisLimits(ImAxis_Y2, 0, 100);
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y1);
          ImPlot::PlotLine("Loss", &g_progress.losses[0], g_progress.losses.size());
          ImPlot::SetAxes(ImAxis_X1, ImAxis_Y2);
          ImPlot::PlotLine("Corrects", &g_progress.corrects[0], g_progress.corrects.size());
          ImPlot::EndPlot();
        }
        
        ImGui::End();
      }

        
      // Rendering
      ImGui::Render();
      int display_w, display_h;
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window);
      auto newhyper = make_shared<HyperParameters>(hp);
      g_hyper.swap(newhyper);
    }

  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();


  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
