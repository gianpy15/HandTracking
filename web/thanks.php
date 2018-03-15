<html>
   <head>
      <title>Hand Tracking</title>
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
      <link rel="stylesheet" type="text/css" href="css/style.css">
   </head>

      <?php
        include 'local_paths.php';
        $script = "source/php_called_scripts/contributors.py";
        $args = "2>&1";
        $cmd = $python_interpreter." ".$script_base.$script." ".$args;
        $out = exec($cmd, $rets, $errorcode);
      ?>

   <body>
       <nav class="navbar navbar-expand-md navbar-dark fixed-top bg-dark">
         <a class="navbar-brand" href="#">HandTracking</a>
         <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarsExampleDefault" aria-controls="navbarsExampleDefault" aria-expanded="false" aria-label="Toggle navigation">
           <span class="navbar-toggler-icon"></span>
         </button>

         <div class="collapse navbar-collapse" id="navbarsExampleDefault">
           <ul class="navbar-nav mr-auto">
             <li class="nav-item active">
               <a class="nav-link" href="./index.html">Home <span class="sr-only">(current)</span></a>
             </li>
           </ul>
         </div>
       </nav>

       <div class="jumbotron">
         <div class="container">
           <h1 class="display-3">Thanks for participating</h1>
           <p>Your help was precious.</p>
           <p> Machine learning needs a huge amount of data, so now you can get another sample, if you want, or just leave the page. ;)</p>
           <p>
               <button class="btn btn-primary btn-lg" role="button" id="get_data">Get a new sample &raquo;</button>
           </p>
             <script>
                 let button = document.getElementById("get_data");
                 let url = new URL(window.location.href);
                 let nick = url.searchParams.get("nick");

                 let new_url = "prepare_data.php?nick=" + nick;

                 button.onclick = () => location.href = new_url;
             </script>
         </div>
       </div>

       <div class="container">
           <div class="col-md-12">
               <a name="top_contributors"></a>
               <hr class="featurette-divider">

                    <div class="row featurette">
                        <div class="col-md-12">
                            <h2 class="featurette-heading" align="center">Top contributors.<br/><br/></h2>
                                <div class="row">
                                    <div class="col-md-6">
                                        <p class="lead" id="first_contributors">
                                            <?php echo $rets[0]; ?><br/>
                                            <?php echo $rets[1]; ?><br/>
                                            <?php echo $rets[2]; ?><br/>
                                            <?php echo $rets[3]; ?><br/>
                                            <?php echo $rets[4]; ?><br/>
                                            <?php echo $rets[5]; ?><br/>
                                            <?php echo $rets[6]; ?><br/>
                                            <?php echo $rets[7]; ?><br/>
                                        </p>
                                    </div>
                                    <div class="col-md-6">
                                        <p class="lead" id="second_contributors">
                                            <?php echo $rets[8]; ?><br/>
                                            <?php echo $rets[9]; ?><br/>
                                            <?php echo $rets[10]; ?><br/>
                                            <?php echo $rets[11]; ?><br/>
                                            <?php echo $rets[12]; ?><br/>
                                            <?php echo $rets[13]; ?><br/>
                                            <?php echo $rets[14]; ?><br/>
                                            <?php echo $rets[15]; ?><br/>
                                        </p>
                                    </div>
                                </div>
                        </div>
                    </div>
                    <hr class="featurette-divider">
           </div>
       </div>



      <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
   </body>
</html>
