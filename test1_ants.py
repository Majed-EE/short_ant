import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import time as T



scene_no=0
scene_xml=["gripper_ants_soft.xml","gripper_ants_rigid.xml"]

xml_path = scene_xml[scene_no]

step_size = 100 
print_camera_config = 0 #set to 1 to print camera config
                        
simend=step_size
#################################### Callback Config Enable to Move Camera ###########################################

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# def init_controller(model,data):
    
#     pass

# def controller(model, data):
    
#     pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)


######################################## Mujoco Mdoel and Data



model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data

######################################### Camera Configuration ######################3

cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = 0.19999999999998996 ; 
cam.elevation = -8.000000000000016 ; 
cam.distance =  2.332639554616304
cam.lookat =np.array([ 0.0 , 0.0 , 0.0 ])


###################################### body data #################################
#vbasic model information using model class

print("Number of bodies:", model.nbody)
print("Number of joints:", model.njnt)
print("Number of DOFs:", model.nv)
print("Number of geoms:", model.ngeom)

# joint dict to store joint info

joint_dict={}
joint_list=[]
for i in range(model.nbody):
    
    body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
    
    print(f"Body {i} name: {body_name}")

#  joint information
for i in range(model.njnt):
    
    joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
    print(f"Joint {i} name: {joint_name}")
    jnt_type=model.jnt_type[i]
    jnt_axis=model.jnt_axis[i]
    print(jnt_axis)
    print(type(jnt_type))
    print(type(jnt_axis))
    print(f"Joint {i} type: {jnt_type}")
    print(f"Joint {i} axis: {jnt_axis}")
    print(f"Joint {i} axis: {jnt_type}")

    joint_dict[str(i)]={"joint_name":joint_name,
                       "joint_type":int(jnt_type),
                       "joint_axis":jnt_axis.tolist()}   # str i where i is the joint id

#geom information
for i in range(model.ngeom):
    geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
    print(f"Geom {i} name: {geom_name}")
    print(f"Geom {i} type: {model.geom_type[i]}")
    print(f"Geom {i} size: {model.geom_size[i]}")

# actuator information
for i in range(model.nu):
    actuator_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"Actuator {i} name: {actuator_name}")
    print(f"Actuator {i} gainprm: {model.actuator_gainprm[i]}")

# sensor information- add tactile sensor, explore which sensor can be added
for i in range(model.nsensordata):
    sensor_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
    print(f"Sensor {i} name: {sensor_name}")
    print(f"Sensor {i} type: {model.sensor_type[i]}")





################################## Controller setup ##################################

# initialize the controller
# init_controller(model,data)


def set_position_servo(joint_id,mov_pos): 
    # isint joint id with actuator is a necessity for this? most likely the case is, 
    # joint id that is passed has an actuator attached to that joint


    
    # model.actuator_gainprm[actuator_no,0]=kp
    # model.actuator_biasprm[actuator_no,1]=-kp
    # print(f"joint id {joint_id}")
    # print(joint_id)
    data.ctrl[1]= 0   # 1 means up # actuator part: how is it moving
    data.ctrl[0]=0    # -1 means squeeze
    print("ctrl matrix")
    print(data.ctrl)
    print(data.act)





def controller(model,data):
    global joint_id_list, joint_dict
    set_position_servo(0,0.5)
    # print("controller initialized")





#set the controller
mj.set_mjcb_control(controller)






#################################### Main Simulation loop #################################

for x in range(simend):
    step_start = T.time()
    mj.mj_step(model, data)
    joint_angles = np.copy(data.qpos)
  
    joint_torques = np.copy(data.qfrc_actuator)
    actuator_force=np.copy(data.qfrc_smooth)
    # data_check_loop=np.copy(data_to_check)
    actuator_force=np.copy(data.actuator_force)
    qfrc_inv_array=np.copy(data.qfrc_inverse)
    print("joint angles")
    print(joint_angles)
    print("actuator force")
    print(actuator_force)

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    #print camera configuration (help to initialize the view)
    if (print_camera_config==1):
        print('cam.azimuth =',cam.azimuth,';','cam.elevation =',cam.elevation,';','cam.distance = ',cam.distance)
        print('cam.lookat =np.array([',cam.lookat[0],',',cam.lookat[1],',',cam.lookat[2],'])')

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
