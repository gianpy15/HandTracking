try {
   window.onload = () => loadImages();
} catch (error){
   alert("error");
}


function loadImages() {
   try{
	   var l_canvas = document.getElementById("left_c");
      var r_canvas = document.getElementById("right_c");
	   var l_ctx = l_canvas.getContext("2d");
      var r_ctx = r_canvas.getContext("2d");

	   var left_img_path = "http://clipartbarn.com/wp-content/uploads/2016/10/Hand-clip-art-free-clipart-images.jpg";
	   var other_url = 'https://cdn.sstatic.net/stackexchange/img/logos/so/so-icon.png'
	   var left_img = new Image();
      var right_img = new Image();

	   right_img.onload = () => r_ctx.drawImage(right_img, 0, 0, r_canvas.width, r_canvas.height);
      left_img.onload = () => l_ctx.drawImage(left_img, 0, 0, l_canvas.width, l_canvas.height);

	   left_img.src = other_url;
      right_img.src = left_img_path;
    }catch(error){
		alert(error)
	}
}
