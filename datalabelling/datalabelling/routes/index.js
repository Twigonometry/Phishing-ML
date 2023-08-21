var express = require('express');
var router = express.Router();
const EmailsModel = require("../models/email");

/* GET home page. */
router.get('/', async function(req, res, next) {
  try {
    var email = await(EmailsModel.findOne({rating: null}).exec());
    res.render('index', { title: 'Data Labelling Portal' });
  } catch (error) {
    return next(error);
  }
});

module.exports = router;
