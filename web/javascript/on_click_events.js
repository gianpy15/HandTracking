function submit_and_exit() {
    if(pinpointer.joints.length === target_joints.length) {
        let data = get_data(pinpointer.joints);

        let params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        function action(resp){
             let nick_append = (nickname != "") ? "?nick="+nickname : "";
             if (resp != "OK") {
                 console.log(resp);
             }
             location.href = "thanks.html"+nick_append;
        }
        sendPost("submit_labels.php", params, action);
    }
    else{
        textManager.resetText();
        textManager.addWarning("Please fill all the points before submission");
    }
}

function submit_and_next_frame(){
    if(pinpointer.joints.length === target_joints.length) {
        let data = get_data(pinpointer.joints);

        let params = "labels="+data.toString()+"&framename="+target_img_url+"&nick="+nickname;
        function loadnew(resp){
            target_img_url = resp;

            helperhand.reset();
            pinpointer.resetJoints();
            console.log(resp);
            pinpointer.setBkgUrl(resp);
            textManager.resetText();
            function setupIndexInfo(resp){
                index_content = parseIndexContent(resp);
                let count = 0.0;
                for(let i=0; i<index_content.length; i++)
                    if(index_content[i] === 1)
                        count += 1.0;
                completion_rate = count / index_content.length;
                textManager.resetText();
            }
            sendPost("read_index.php", "framename="+target_img_url, setupIndexInfo);
        }
        textManager.resetText();
        textManager.addNotice("Loading the new image...");
        document.getElementById("img_loader").style.visibility = "visible";
        sendPost("next_frame.php", '', loadnew);
        sendPost("submit_labels.php", params, (r) => console.log(r));
    }
    else{
        textManager.resetText();
        textManager.addWarning("Please fill all the points before submission");
    }
}


function sendPost(destination, params='', responseAction=(resp) => null){
    let request = new XMLHttpRequest();
    request.open('POST', destination, true);
    request.onreadystatechange = function () {
        if(request.readyState === XMLHttpRequest.DONE && request.status === 200){
            responseAction(request.responseText);
        }
    };
    request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    request.send(params);
}

function leavePage(){
    alert("leaving");
    sendPost("leaving.php", "framename="+target_img_url, (r) => alert(r));
}

function get_data(points) {
   let ret = [];
   for (let i = 0; i < points.length; i++) {
      ret.push(points[i].px);
      ret.push(points[i].py);
      ret.push(points[i].occluded);
   }
   return ret;
}
