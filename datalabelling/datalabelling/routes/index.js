var express = require('express');
var router = express.Router();
const EmailsModel = require("../models/email");
const mongoose = require("mongoose");

/* GET home page. */
router.get('/', async function(req, res, next) {
  try {
    var email = await(EmailsModel.findOne({rating: null}).exec());
    console.log(email);
    res.render('index', { title: 'Data Labelling Portal', email: email });
  } catch (error) {
    return next(error);
  }
});

router.post('/rate', async function(req, res, next) {
  try {
    const params = req.body;
    const id = new mongoose.Types.ObjectId(params.id);
    console.log(id);

    const tmp_rating = {
      authoritative: (params.authoritative == "on" ? 1 : 0),
      threatening: (params.threatening == "on" ? 1 : 0),
      rewarding: (params.rewarding == "on" ? 1 : 0),
      unnatural: (params.unnatural == "on" ? 1 : 0),
      emotional: (params.emotional == "on" ? 1 : 0),
      provoking: (params.provoking == "on" ? 1 : 0),
      timesensitive: (params.timesensitive == "on" ? 1 : 0),
      imperative: (params.imperative == "on" ? 1 : 0)
    };

    console.log(tmp_rating);

    await EmailsModel.findByIdAndUpdate(id, {rating: tmp_rating});

    res.redirect('/')
  } catch (error) {
    console.log(error);
    return next(error);
  }
})

module.exports = router;
