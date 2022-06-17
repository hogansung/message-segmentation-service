import axios from "axios";
import React from 'react';

class QueryFormAndResponse extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            input_string: '',
            optimize_segments_checkbox: false,
            response_entities: []
        };

        this.handleInputStringChange = this.handleInputStringChange.bind(this);
        this.handleOptimizeSegmentsCheckboxChange = this.handleOptimizeSegmentsCheckboxChange.bind(this);
        this.handleSubmit = this.handleSubmit.bind(this);
    }

    handleInputStringChange(event) {
        this.setState({input_string: event.target.value});
    }

    handleOptimizeSegmentsCheckboxChange(event) {
        this.setState({optimize_segments_checkbox: event.target.checked});
        console.log(this.state);
    }

    handleSubmit(event) {
        console.log(this.state);
        axios.post("/query", {
            "message": this.state.input_string,
            "b_optimized": this.state.optimize_segments_checkbox
        })
            .then(response => {
                this.setState({response_entities: response.data.response_entities});
            }).catch((error) => {
                if (error.response) {
                    console.log(error.response)
                    console.log(error.response.status)
                    console.log(error.response.headers)
                }
            })

        event.preventDefault();
    }

    render() {
        return (
            <div className="bd-content ps-lg-2">
                <div>
                    <form onSubmit={this.handleSubmit}>
                        <div className="mb-3">
                            <label htmlFor="inputString" className="form-label"><b>Input String</b></label>
                            <input type="text" className="form-control" id="inputString" autoComplete="off"
                                spellCheck="false" onChange={this.handleInputStringChange}/>
                        </div>
                        <div className="form-check">
                            <input type="checkbox" className="form-check-input" id="optimizeSegmentsCheckbox"
                                onClick={this.handleOptimizeSegmentsCheckboxChange.bind(this)}/>
                            <label className="form-check-label" htmlFor="optimizeSegmentsCheckbox">Optimize Segments</label>
                        </div>
                        <br/>
                        <button type="submit" className="btn btn-primary">Submit</button>
                    </form>
                </div>
                <br/>
                <br/>
                <br/>
                <table className="table table-hover">
                    <thead>
                        <tr>
                            <th scope="col">#</th>
                            <th scope="col">Segment</th>
                            <th scope="col">Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        {
                            this.state.response_entities.map((response_entity, index) => (
                                <tr key={index}>
                                    <th scope="row">{index}</th>
                                    <td>{response_entity.segment}</td>
                                    <td>{response_entity.score}</td>
                                </tr>
                            ))
                        }
                    </tbody>
                </table>
            </div>
        );
    }
}

export default QueryFormAndResponse;
