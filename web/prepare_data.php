<html>
   <head>
      <title>Your contribute</title>
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
      <link rel="stylesheet" type="text/css" href="css/style.css">
   </head>

   <?php
        include 'local_paths.php';
        $script = "source/php_called_scripts/get_frame.py";
        $args = "2>&1";
        $cmd = $python_interpreter." ".$script_base.$script." ".$args;
        $out = system($cmd, $errorcode);
        if ($errorcode == 0){
            $error = "";
            $imgurl = $out;
        }else{
            $error = $out;
        }
        $nick = $_GET["nick"]
   ?>

   <body>
       <nav class="navbar navbar-expand-md navbar-dark fixed-top bg-dark">
         <a class="navbar-brand" href="#">HandTracking</a>
         <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarsExampleDefault" aria-controls="navbarsExampleDefault" aria-expanded="false" aria-label="Toggle navigation">
           <span class="navbar-toggler-icon"></span>
         </button>

         <div class="collapse navbar-collapse" id="navbarsExampleDefault">
           <ul class="navbar-nav mr-auto">
           </ul>
         </div>
       </nav>

       <div class="jumbotron">
         <div class="container">
           <h1>Let's begin</h1>
           <p>Here you're asked to set a green point on every visible junction and a red one on every non-visible junction</p>
         </div>
       </div>
       <div class="container">
             <div class="row">
                 <div class="col md-auto">
                     <canvas id="left_c" width="500px" height="500px"></canvas>
                     <canvas id="left_c_points" width="500px" height="500px"></canvas>
                  </div>
                  <div class="col md-auto">
                      <canvas id="right_c" width="500px" height="500px"></canvas>
                      <canvas id="right_c_points" width="500px" height="500px"></canvas>
                  </div>
             </div>
             <div>
                 <button id="undo_b">Undo</button>
                 <!--<a id="undo_b" class="btn btn-primary btn-sm" href="" role="button">Undo</a>-->
                 <button id="new_sample_b">Submit and get other sample</button>
                 <!--<a id="new_sample_b" class="btn btn-primary btn-sm" href="" role="button">Submit and get other sample</a>-->
                 <button id="submit_b">Submit and exit</button>
                 <!--<a id="submit_b" class="btn btn-primary btn-sm" href="" role="button">Submit and exit</a>-->
             </div>
       </div>

       <script src="javascript/points.js" type="text/javascript"></script>
       <script type="text/javascript">
            var nickname = "<?php echo $nick; ?>";
            var sample_img_url = "http://dreamicus.com/data/hand/hand-07.jpg";
            var target_img_url = "<?php echo $imgurl; ?>";
            var target_joints = [new Point(0.3, 0.3, false), new Point(0.5, 0.5, false), new Point(0.7, 0.7, false)];
       </script>
       <?php echo "Nickname: ".$nick; ?>
       <script src="javascript/hand_tracking_samples.js" type="text/javascript"></script>
      <script src="javascript/on_click_events.js" type="text/javascript"></script>
      <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
   </body>
</html>
